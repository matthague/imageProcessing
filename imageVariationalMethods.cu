#include "imageVariationalMethods.cuh"
#include <stdint.h>

#define THREADS_PER_BLOCK 32
#define STABILITY_EPSILON 0.01

// the fast inverse sqrt algorithm adapted for doubles and cuda
// https://en.wikipedia.org/wiki/Fast_inverse_square_root
__device__ double fast_inverse_sqrt(double number) {
    uint64_t i;
    double x2, y;
    const double threehalfs = 1.5F;

    x2 = number * 0.5F;
    y = number;
    memcpy(&i, &y, sizeof(y));
    i = 0x5FE6EB50C7B537A9 - (i >> 1); // magic constant
    memcpy(&y, &i, sizeof(y));
    y = y * (threehalfs - (x2 * y * y));   // 1st iteration
    y = y * (threehalfs - (x2 * y * y));   // 2nd iteration, this can be removed

    return y;
}

// calculates the element-wise inverse square root of a cube
// result[i] = 1/((a[i]^3)^.5)
__global__ void inverse_three_halves(double *result, double *a, double epsilon, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        double f = a[idx] + epsilon; // adding epsilon avoids a numerical singularity
        f = f * f * f; // cube step
        result[idx] = fast_inverse_sqrt(f); // inverse square root step
    }
}

// calculates centered difference approximation to second partial derivative of u with respect to x
__global__ void calc_ux(double *u_x, double *u, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int column = idx % width;
        double u_left;
        double u_right;

        if (column == 0) {
            u_right = u[idx + 1];
            u_left = u[idx + 1];
        } else if (column == width - 1) {
            u_right = u[idx - 1];
            u_left = u[idx - 1];
        } else {
            u_right = u[idx + 1];
            u_left = u[idx - 1];
        }

        u_x[idx] = (u_right - u_left) / 2;
    }
}

// calculates centered difference approximation to partial derivative of u with respect to y
__global__ void calc_uy(double *u_y, double *u, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int column = idx % width;
        int row = (idx - column) / width;
        double u_up;
        double u_down;

        if (row == 0) {
            u_up = u[idx + width];
            u_down = u[idx + width];
        } else if (row == height - 1) {
            u_up = u[idx - width];
            u_down = u[idx - width];
        } else {
            u_up = u[idx - width];
            u_down = u[idx + width];
        }

        u_y[idx] = (u_up - u_down) / 2;
    }
}

// calculates centered difference approximation to second partial derivative of u with respect to x
__global__ void calc_uxx(double *u_xx, double *u, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int column = idx % width;
        double u_left;
        double u_right;
        double u_center = u[idx];

        if (column == 0) {
            u_right = u[idx + 1];
            u_left = u[idx + 1];
        } else if (column == width - 1) {
            u_right = u[idx - 1];
            u_left = u[idx - 1];
        } else {
            u_right = u[idx + 1];
            u_left = u[idx - 1];
        }

        u_xx[idx] = u_right + u_left - 2 * u_center;
    }
}

// calculates centered difference approximation to second partial derivative of u with respect to y
__global__ void calc_uyy(double *u_yy, double *u, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int column = idx % width;
        int row = (idx - column) / width;
        double u_up;
        double u_down;
        double u_center = u[idx];

        if (row == 0) {
            u_up = u[idx + width];
            u_down = u[idx + width];
        } else if (row == height - 1) {
            u_up = u[idx - width];
            u_down = u[idx - width];
        } else {
            u_up = u[idx - width];
            u_down = u[idx + width];
        }

        u_yy[idx] = u_up + u_down - 2 * u_center;
    }
}

// calculates result[i] = (s1 * a[i]) + (s2 * b[i]) for valid i
__global__ void scaledMatrixSum(double *result, double *a, double *b, double s1, double s2, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        result[idx] = (s1 * a[idx]) + (s2 * b[idx]);
    }
}

// calculates result[i] = scalar * a[i] * b[i] for valid i
__global__ void
scaledMultiplyMatrixEntries(double *result, double *a, double *b, double scalar, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        result[idx] = scalar * a[idx] * b[idx];
    }
}

// performs variational diffusion on an image
void imageVariationalMethods::diffusion(double **pixelArrays, int width, int height, int depth, double deltaTime,
                                        double lambda, int numSteps) {
    int totalThreads = width * height;
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // setup the device matrices that we will need
    double *f;
    double *u_curr;
    double *u_t;
    double *u_xx;
    double *u_yy;
    cudaMalloc(&f, width * height * sizeof(double));
    cudaMalloc(&u_curr, width * height * sizeof(double));
    cudaMalloc(&u_t, width * height * sizeof(double));
    cudaMalloc(&u_xx, width * height * sizeof(double));
    cudaMalloc(&u_yy, width * height * sizeof(double));

    // perform variational diffusion on each image plane (see paper/report for math details)
    for (int k = 0; k < depth; k++) {
        // copy inputs to device
        cudaMemcpy(f, pixelArrays[k], width * height * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(u_curr, pixelArrays[k], width * height * sizeof(double), cudaMemcpyHostToDevice);

        // evolve the gradient descent flow
        for (int i = 0; i < numSteps; i++) {
            calc_uxx<<<numBlocks, blockSize>>>(u_xx, u_curr, width, height);
            calc_uyy<<<numBlocks, blockSize>>>(u_yy, u_curr, width, height);
            cudaDeviceSynchronize();

            // u_t = u_xx + u_yy - the laplacian of u
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_t, u_xx, u_yy, 1.0, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_t = f + lambda * u_t
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_t, f, u_t, 1.0, lambda, width, height);
            cudaDeviceSynchronize();

            // u_t = -u + u_t
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_t, u_curr, u_t, -1.0, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_curr = deltat * u_t + u_curr
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_curr, u_t, u_curr, deltaTime, 1.0, width, height);
            cudaDeviceSynchronize();
        }

        // copy results out
        cudaMemcpy(pixelArrays[k], u_curr, width * height * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // free resources
    cudaFree(f);
    cudaFree(u_curr);
    cudaFree(u_t);
    cudaFree(u_xx);
    cudaFree(u_yy);
}

// performs a total variational reduction on an image
void imageVariationalMethods::total(double **pixelArrays, int width, int height, int depth, double deltaTime,
                                    double lambda, int numSteps) {
    int totalThreads = width * height;
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // setup the device matrices that we will need
    double *f;
    double *u_curr;
    double *u_t;
    double *u_x;
    double *u_y;
    double *u_xy;
    double *u_xx;
    double *u_yy;

    cudaMalloc(&f, width * height * sizeof(double));
    cudaMalloc(&u_curr, width * height * sizeof(double));
    cudaMalloc(&u_t, width * height * sizeof(double));
    cudaMalloc(&u_x, width * height * sizeof(double));
    cudaMalloc(&u_y, width * height * sizeof(double));
    cudaMalloc(&u_xy, width * height * sizeof(double));
    cudaMalloc(&u_xx, width * height * sizeof(double));
    cudaMalloc(&u_yy, width * height * sizeof(double));

    // perform total variational descent on each image plane (see paper/report for math details)
    for (int k = 0; k < depth; k++) {
        // copy inputs to device
        cudaMemcpy(f, pixelArrays[k], width * height * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(u_curr, pixelArrays[k], width * height * sizeof(double), cudaMemcpyHostToDevice);

        // evolve the gradient descent flow
        for (int i = 0; i < numSteps; i++) {
            // calculate pure partials
            calc_uxx<<<numBlocks, blockSize>>>(u_xx, u_curr, width, height);
            calc_uyy<<<numBlocks, blockSize>>>(u_yy, u_curr, width, height);
            calc_uy<<<numBlocks, blockSize>>>(u_y, u_curr, width, height);
            calc_ux<<<numBlocks, blockSize>>>(u_x, u_curr, width, height);
            cudaDeviceSynchronize();

            // calculate mixed partial
            calc_uy<<<numBlocks, blockSize>>>(u_xy, u_x, width, height);
            cudaDeviceSynchronize();

            // u_xy = -2 * u_xy * u_x * u_y (step 1)
            scaledMultiplyMatrixEntries<<<numBlocks, blockSize>>>(u_xy, u_xy, u_x, -2.0, width, height);
            cudaDeviceSynchronize();
            scaledMultiplyMatrixEntries<<<numBlocks, blockSize>>>(u_xy, u_xy, u_y, 1, width, height);
            cudaDeviceSynchronize();

            // u_x = u_x^2, u_y = y_y^2 (step 2)
            scaledMultiplyMatrixEntries<<<numBlocks, blockSize>>>(u_y, u_y, u_y, 1.0, width, height);
            scaledMultiplyMatrixEntries<<<numBlocks, blockSize>>>(u_x, u_x, u_x, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_yy = u_yy * u_x^2, u_xx = u_xx * u_y^2 (step 3)
            scaledMultiplyMatrixEntries<<<numBlocks, blockSize>>>(u_yy, u_yy, u_x, 1.0, width, height);
            scaledMultiplyMatrixEntries<<<numBlocks, blockSize>>>(u_xx, u_xx, u_y, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_xy = u_xy + u_yy + u_xx forming the top part of the big eq... (step 4)
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_xy, u_xy, u_yy, 1.0, 1.0, width, height);
            cudaDeviceSynchronize();
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_xy, u_xy, u_xx, 1.0, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_x = u_x^2 + u_y^2 (step 5)
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_x, u_x, u_y, 1.0, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_x = inverse three halves (step 6)
            inverse_three_halves<<<numBlocks, blockSize>>>(u_x, u_x, STABILITY_EPSILON, width, height);
            cudaDeviceSynchronize();

            // u_x = u_x * u_xy (step 7)
            scaledMultiplyMatrixEntries<<<numBlocks, blockSize>>>(u_x, u_xy, u_x, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_y = -lambda*(u_curr - f) (step 8)
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_y, u_curr, f, -lambda, lambda, width, height);
            cudaDeviceSynchronize();

            // u_t = u_y + u_x (step 9)
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_t, u_x, u_y, 1.0, 1.0, width, height);
            cudaDeviceSynchronize();

            // u_curr = deltat*u_t + u_curr
            scaledMatrixSum<<<numBlocks, blockSize>>>(u_curr, u_t, u_curr, deltaTime, 1.0, width, height);
            cudaDeviceSynchronize();
        }

        // copy results out
        cudaMemcpy(pixelArrays[k], u_curr, width * height * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // free resources
    cudaFree(f);
    cudaFree(u_curr);
    cudaFree(u_t);
    cudaFree(u_x);
    cudaFree(u_y);
    cudaFree(u_xy);
    cudaFree(u_xx);
    cudaFree(u_yy);
}