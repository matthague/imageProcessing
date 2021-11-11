#include "imageFilters.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cufft.h>

#define THREADS_PER_BLOCK 32

typedef double2 Complex;

__global__ void lowPassKernel(cufftDoubleComplex *complexImage, int width, int height, float cutoff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int x = idx % width;
        int y = (idx - x) / width;

        float u = (x - (width / 2.0)) / ((float) width);
        float v = (y - (height / 2.0)) / ((float) height);
        float f = cutoff;

        f *= f;
        u *= u;
        v *= v;

        // likes f in range 0.0 -> .5 *(2)**.5
        if ((u + v) < f) {
            complexImage[idx].x = 0;
            complexImage[idx].y = 0;
        }
    }
}

void imageFilters::fourierLowPass(double **pixelArrays, int width, int height, int depth, float cutoff) {
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = ((width * height) + blockSize - 1) / blockSize;

    float effectiveCutoff = cutoff * (0.70710678118); // scales correctly for transformed image coordinates
    Complex *complexImage = new Complex[width * height];
    cufftDoubleComplex *deviceComplexImage;
    cudaMalloc(&deviceComplexImage, width * height * sizeof(Complex));

    // setup handle and plan
    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CUFFT_Z2Z);

    for (int k = 0; k < depth; k++) {

        // load (shifted) image as complex
        for (int i = 0; i < width * height; i++) {
            complexImage[i].x = pixelArrays[k][i] - 128.0;
            complexImage[i].y = 0;
        }

        // copy input to the device
        cudaMemcpy(deviceComplexImage, complexImage, width * height * sizeof(Complex), cudaMemcpyHostToDevice);

        // compute the forward 2d fft
        cufftExecZ2Z(plan, (cufftDoubleComplex *) deviceComplexImage, (cufftDoubleComplex *) deviceComplexImage,
                     CUFFT_FORWARD);
        cudaDeviceSynchronize();

        // run the low pass mask
        lowPassKernel<<<numBlocks, blockSize>>>(deviceComplexImage, width, height, effectiveCutoff);
        cudaDeviceSynchronize();

        // compute the backward 2d fft
        cufftExecZ2Z(plan, (cufftDoubleComplex *) deviceComplexImage, (cufftDoubleComplex *) deviceComplexImage,
                     CUFFT_INVERSE);
        cudaDeviceSynchronize();

        // copy the result back
        cudaMemcpy(complexImage, deviceComplexImage, width * height * sizeof(Complex), cudaMemcpyDeviceToHost);

        // save the normalized result (unshifted)
        for (int i = 0; i < width * height; i++) {
            pixelArrays[k][i] = (complexImage[i].x / (width * height)) + 128.0;
        }

    }

    delete complexImage;
    cudaFree(deviceComplexImage);
    cufftDestroy(plan);
}

__global__ void medianKernel(double *inputImage, double *outputImage, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int column = idx % width;
        int row = (idx - column) / width;

        const int neighborhoodSize = 9;
        const int neighborhoodWidth = 3;
        double neighborhood[neighborhoodSize] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        if ((row != 0) && (column != 0) && (row != height - 1) && (column != width - 1)) {

            // fill the neighborhood
            for (int x = 0; x < neighborhoodWidth; x++) {
                for (int y = 0; y < neighborhoodWidth; y++) {
                    neighborhood[x * neighborhoodWidth + y] = inputImage[(row + x - 1) * width + (column + y - 1)];
                }
            }

            // sort the neighborhood
            for (int i = 0; i < neighborhoodSize; i++) {
                for (int j = i + 1; j < neighborhoodSize; j++) {
                    if (neighborhood[i] > neighborhood[j]) {
                        // swap entries
                        double tmp = neighborhood[i];
                        neighborhood[i] = neighborhood[j];
                        neighborhood[j] = tmp;
                    }
                }
            }

            // set entry to median
            outputImage[idx] = neighborhood[4];
        } else {
            outputImage[idx] = inputImage[idx];
        }
    }
}

void imageFilters::median(double **pixelArrays, int width, int height, int depth, int passes) {
    // kernel launch parameters
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = ((width * height) + blockSize - 1) / blockSize;

    // allocate memory for images on device
    double *deviceInputImage;
    double *deviceResultImage;
    cudaMalloc(&deviceInputImage, width * height * sizeof(*deviceInputImage));
    cudaMalloc(&deviceResultImage, width * height * sizeof(*deviceResultImage));

    // process each color plane
    for (int k = 0; k < depth; k++) {
        cudaMemcpy(deviceInputImage, pixelArrays[k], width * height * sizeof(*deviceInputImage),
                   cudaMemcpyHostToDevice);
        for (int i = 0; i < passes; i++) {
            medianKernel<<<numBlocks, blockSize>>>(deviceInputImage, deviceResultImage, width, height);
            cudaDeviceSynchronize();
            cudaMemcpy(deviceInputImage, deviceResultImage, width * height * sizeof(*deviceResultImage),
                       cudaMemcpyDeviceToDevice);
        }
        cudaMemcpy(pixelArrays[k], deviceResultImage, width * height * sizeof(*deviceResultImage),
                   cudaMemcpyDeviceToHost);
    }

    // free resources
    cudaFree(deviceInputImage);
    cudaFree(deviceResultImage);
}