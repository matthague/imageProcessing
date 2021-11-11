#include "imageAlgebra.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void imageAlgebra::kSVD(double **pixelArrays, int width, int height, int depth, int k) {
    // set up solver handles
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    // dimensions for column major matrix (autotransposed so m > n)
    int m;
    int n;

    if (height >= width) {
        m = height;
        n = width;
    } else {
        m = width;
        n = height;
    }

    int lda = m;

    // setup host arrays
    double *A = NULL; // no need to malloc, just set to color plane pointer
    double *U = NULL; // [lda * m] n-by-m left eigenvectors
    double *VT = NULL; // [lda * n] n-by-n unitary matrix
    double *S = NULL; // [n] singular values

    cudaMallocHost(&U, lda * m * (sizeof(*U)));
    cudaMallocHost(&VT, lda * n * (sizeof(*VT)));
    cudaMallocHost(&S, n * (sizeof(*S)));

    // setup device arrays
    double *d_A = NULL;
    double *d_S = NULL;
    double *d_U = NULL;
    double *d_VT = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    double *d_rwork = NULL;
    double *d_W = NULL;  // W = S*VT

    cudaMalloc(&d_A, sizeof(double) * lda * n);
    cudaMalloc(&d_S, sizeof(double) * n);
    cudaMalloc(&d_U, sizeof(double) * lda * m);
    cudaMalloc(&d_VT, sizeof(double) * lda * n);
    cudaMalloc(&devInfo, sizeof(int));
    cudaMalloc(&d_W, sizeof(double) * lda * n);

    // compute svd on each color plane
    for (int i = 0; i < depth; i++) {
        A = pixelArrays[i];

        int lwork = 0;
        const double h_one = 1;
        const double h_zero = 0;

        // copy input array to device
        cudaMemcpy(d_A, A, sizeof(double) * lda * n, cudaMemcpyHostToDevice);

        // query working space of SVD solver
        cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);
        cudaMalloc(&d_work, sizeof(double) * lwork);

        // compute SVD
        signed char jobu = 'A'; // all m columns of U
        signed char jobvt = 'A'; // all n columns of VT
        cusolverDnDgesvd(
                cusolverH,
                jobu,
                jobvt,
                m,
                n,
                d_A,
                lda,
                d_S,
                d_U,
                lda,  // ldu
                d_VT,
                lda, // ldvt,
                d_work,
                lwork,
                d_rwork,
                devInfo);

        cudaDeviceSynchronize();

        cudaFree(d_work);

        // copy singular value results back
        cudaMemcpy(S, d_S, sizeof(double) * n, cudaMemcpyDeviceToHost);

        // save only data at indicies where singular value is in top k
        for (int j = 0; j < n; j++) {
            if (j >= k) {
                S[j] = 0.0;
            }
        }
        cudaMemcpy(d_S, S, sizeof(double) * n, cudaMemcpyHostToDevice);

        // W = S*VT
        cublasDdgmm(
                cublasH,
                CUBLAS_SIDE_LEFT,
                n,
                n,
                d_VT,
                lda,
                d_S,
                1,
                d_W,
                lda);

        cudaDeviceSynchronize();

        // A = U*W
        cublasDgemm_v2(
                cublasH,
                CUBLAS_OP_N, // U
                CUBLAS_OP_N, // W
                m, // number of rows of A
                n, // number of columns of A
                n, // number of columns of U
                &h_one, // host pointer
                d_U, // U
                lda,
                d_W, // W
                lda,
                &h_zero, // host pointer
                d_A,
                lda);

        cudaDeviceSynchronize();

        // copy results back
        cudaMemcpy(A, d_A, sizeof(double) * lda * n, cudaMemcpyDeviceToHost);
    }

    // free resources
    cudaFree(d_A);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_W);

    cudaFree(devInfo);
    cudaFree(d_rwork);

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    cudaFreeHost(U);
    cudaFreeHost(S);
    cudaFreeHost(VT);
}
