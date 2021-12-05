#include "imageOperations.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>

#define THREADS_PER_BLOCK 32
#define REGULARIZATION_C1 0.0001 * 255.0 * 255.0
#define REGULARIZATION_C2 0.0009 * 255.0 * 255.0

__global__ void initRNGKernel(curandState *state, int seed, int maxIndex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < maxIndex) {
        curand_init(seed, idx, 0, &state[idx]); // seeding is a VERY slow step
    }
}

void imageOperations::initRNG(curandState *state, int stateSize, int seed) {
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (stateSize + blockSize - 1) / blockSize;
    initRNGKernel<<<numBlocks, blockSize>>>(state, seed, stateSize);
    cudaDeviceSynchronize();
}

__global__ void addGaussianNoiseKernel(double *pixelArray, curandState *state, double noiseStdev, int totalThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        pixelArray[idx] += noiseStdev * curand_normal(&state[idx]);
    }
}

// adds white gaussian noise to an image
void imageOperations::addGaussianNoise(double **pixelArrays, curandState *rngState, int width, int height, int depth,
                                       double noiseStdev) {
    int totalThreads = width * height;
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // allocate space for each color plane
    double *devicePixelArray;
    cudaMalloc(&devicePixelArray, totalThreads * sizeof(*devicePixelArray));

    for (int colorPlane = 0; colorPlane < depth; colorPlane++) {
        cudaMemcpy(devicePixelArray, pixelArrays[colorPlane], totalThreads * sizeof(double), cudaMemcpyHostToDevice);

        addGaussianNoiseKernel<<<numBlocks, blockSize>>>(devicePixelArray, rngState, noiseStdev, totalThreads);
        cudaDeviceSynchronize();

        cudaMemcpy(pixelArrays[colorPlane], devicePixelArray, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // free device arrays
    cudaFree(devicePixelArray);
}

__global__ void addSANDPNoiseKernel(double *pixelArray, curandState *state, double noiseRate, int totalThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        float selector = 255.0 * curand_uniform(&state[idx]);
        if (selector < noiseRate) {
            pixelArray[idx] = 255.0 * curand_uniform(&state[idx]);
        }
    }
}

// adds salt and pepper noise to an image
void imageOperations::addSANDPNoise(double **pixelArrays, curandState *rngState, int width, int height, int depth,
                                    double noiseRate) {
    int totalThreads = width * height;
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // allocate space for each color plane
    double *devicePixelArray;
    cudaMalloc(&devicePixelArray, totalThreads * sizeof(*devicePixelArray));

    for (int colorPlane = 0; colorPlane < depth; colorPlane++) {
        cudaMemcpy(devicePixelArray, pixelArrays[colorPlane], totalThreads * sizeof(double), cudaMemcpyHostToDevice);

        addSANDPNoiseKernel<<<numBlocks, blockSize>>>(devicePixelArray, rngState, noiseRate, totalThreads);
        cudaDeviceSynchronize();

        cudaMemcpy(pixelArrays[colorPlane], devicePixelArray, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // free device arrays
    cudaFree(devicePixelArray);
}

__global__ void
squaredDifferenceKernel(double *pixelArrayA, double *pixelArrayB, double *deltaArray, int totalThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        double delta = pixelArrayA[idx] - pixelArrayB[idx];
        deltaArray[idx] = delta * delta;
    }
}

__global__ void sumReduce(double *array, int activeElements, int maxIndex) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    double element1 = 0.0;
    double element2 = 0.0;

    if ((2 * idx < activeElements) && (2 * idx < maxIndex)) {
        element1 = array[2 * idx];
    }
    if ((2 * idx + 1 < activeElements) && (2 * idx < maxIndex)) {
        element2 = array[2 * idx + 1];
    }

    if (idx < maxIndex) {
        array[idx] = element1 + element2;
    }
}

// computes the mean squared error of an image from a target image
double imageOperations::computeMSE(double **pixelArraysA, double **pixelArraysB, int width, int height, int depth) {
    int totalThreads = width * height;
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // the accumulated result
    double result = 0;

    // allocate space for each color plane
    double *devicePixelArrayA;
    double *devicePixelArrayB;
    double *devicePixelArrayDelta;
    cudaMalloc(&devicePixelArrayA, totalThreads * sizeof(*devicePixelArrayA));
    cudaMalloc(&devicePixelArrayB, totalThreads * sizeof(*devicePixelArrayB));
    cudaMalloc(&devicePixelArrayDelta, totalThreads * sizeof(*devicePixelArrayDelta));

    for (int colorPlane = 0; colorPlane < depth; colorPlane++) {
        cudaMemcpy(devicePixelArrayA, pixelArraysA[colorPlane], totalThreads * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(devicePixelArrayB, pixelArraysB[colorPlane], totalThreads * sizeof(double), cudaMemcpyHostToDevice);

        // compute squared differences
        squaredDifferenceKernel<<<numBlocks, blockSize>>>(devicePixelArrayA, devicePixelArrayB, devicePixelArrayDelta,
                                                          totalThreads);
        cudaDeviceSynchronize();

        // a delta array of length totalThreads is now in memory, reduce it so that the first entry is the sum...
        // increase active as a power of 2 until it is larger than totalThreads
        int activeElements = 1;
        while (activeElements <= totalThreads) {
            activeElements = activeElements << 1;
        }

        // sum reduce the elements
        while (activeElements > 1) {
            int reduceThreads = activeElements;
            int reduceBlockSize = THREADS_PER_BLOCK;
            int reduceNumBlocks = (reduceThreads + reduceBlockSize - 1) / reduceBlockSize;
            sumReduce<<<reduceNumBlocks, reduceBlockSize>>>(devicePixelArrayDelta, activeElements, totalThreads);
            cudaDeviceSynchronize();
            activeElements = activeElements >> 1;
        }

        // copy accumulated sum back
        double accumulator;
        cudaMemcpy(&accumulator, devicePixelArrayDelta, 1 * sizeof(double), cudaMemcpyDeviceToHost);

        // add MSE to the result
        result += accumulator;
    }

    // free the arrays
    cudaFree(devicePixelArrayA);
    cudaFree(devicePixelArrayB);
    cudaFree(devicePixelArrayDelta);

    result = result / (width * height * depth);
    return result;
}

// computes the peak signal to noise ratio
double imageOperations::computePSNR(double mse) {
    if (mse == 0.0) {
        return 100.0;
    }
    float psnr = 10.0 * log(255.0 * 255.0 / mse) / log(10.0);
    return psnr;
}

// computes the mean of a list of pixel value
double imageOperations::computeMean(double **pixelArrays, int width, int height, int depth) {
    int totalElements = width * height; // total elements per color plane

    // the accumulated result
    double result = 0;

    double *devicePixelArray;
    cudaMalloc(&devicePixelArray, totalElements * sizeof(*devicePixelArray));

    for (int colorPlane = 0; colorPlane < depth; colorPlane++) {
        cudaMemcpy(devicePixelArray, pixelArrays[colorPlane], totalElements * sizeof(double), cudaMemcpyHostToDevice);

        // increase active as a power of 2 until it is larger than totalElements
        int activeElements = 1;
        while (activeElements <= totalElements) {
            activeElements = activeElements << 1;
        }

        // sum reduce the elements
        while (activeElements > 1) {
            int reduceThreads = activeElements;
            int reduceBlockSize = THREADS_PER_BLOCK;
            int reduceNumBlocks = (reduceThreads + reduceBlockSize - 1) / reduceBlockSize;
            sumReduce<<<reduceNumBlocks, reduceBlockSize>>>(devicePixelArray, activeElements, totalElements);
            cudaDeviceSynchronize();
            activeElements = activeElements >> 1;
        }

        // copy accumulated sum back
        double accumulator;
        cudaMemcpy(&accumulator, devicePixelArray, 1 * sizeof(double), cudaMemcpyDeviceToHost);

        // accumulate plane mean to result
        result += accumulator;
    }

    // free the array
    cudaFree(devicePixelArray);

    result = result / (width * height * depth);
    return result;
}

__global__ void squaredDifferenceSimpleKernel(double *pixelArray, double mean, int totalThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        double delta = pixelArray[idx] - mean;
        pixelArray[idx] = delta * delta;
    }
}

// computes variance of a list of pixel values
double imageOperations::computeVariance(double **pixelArrays, double mean, int width, int height, int depth) {
    int totalThreads = width * height;
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // the accumulated result
    double result = 0;

    double *devicePixelArray;
    cudaMalloc(&devicePixelArray, totalThreads * sizeof(*devicePixelArray));

    for (int colorPlane = 0; colorPlane < depth; colorPlane++) {
        cudaMemcpy(devicePixelArray, pixelArrays[colorPlane], totalThreads * sizeof(double), cudaMemcpyHostToDevice);

        // compute simple squared differences in place
        squaredDifferenceSimpleKernel<<<numBlocks, blockSize>>>(devicePixelArray, mean, totalThreads);
        cudaDeviceSynchronize();

        // a delta array of length totalThreads is now in memory, reduce it so that the first entry is the sum...
        // increase active as a power of 2 until it is larger than totalThreads
        int activeElements = 1;
        while (activeElements <= totalThreads) {
            activeElements = activeElements << 1;
        }

        // sum reduce the elements
        while (activeElements > 1) {
            int reduceThreads = activeElements;
            int reduceBlockSize = THREADS_PER_BLOCK;
            int reduceNumBlocks = (reduceThreads + reduceBlockSize - 1) / reduceBlockSize;
            sumReduce<<<reduceNumBlocks, reduceBlockSize>>>(devicePixelArray, activeElements, totalThreads);
            cudaDeviceSynchronize();
            activeElements = activeElements >> 1;
        }

        // copy accumulated sum back
        double accumulator;
        cudaMemcpy(&accumulator, devicePixelArray, 1 * sizeof(double), cudaMemcpyDeviceToHost);

        // accumulate plane variance to result
        result += accumulator;
    }
    // free the arrays
    cudaFree(devicePixelArray);

    result = result / (width * height * depth);
    return result;
}

__global__ void
covarianceKernel(double *pixelArrayA, double *pixelArrayB, double *deltaArray, double meanA, double meanB,
                 int totalThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        double deltaA = pixelArrayA[idx] - meanA;
        double deltaB = pixelArrayB[idx] - meanB;
        deltaArray[idx] = deltaA * deltaB;
    }
}

// computes pearson covariance
double
imageOperations::computeCovariance(double **pixelArraysA, double **pixelArraysB, double meanA, double meanB, int width,
                                   int height, int depth) {
    int totalThreads = width * height;
    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    // the accumulated result
    double result = 0;

    // allocate space for each color plane
    double *devicePixelArrayA;
    double *devicePixelArrayB;
    double *devicePixelArrayDelta;
    cudaMalloc(&devicePixelArrayA, totalThreads * sizeof(*devicePixelArrayA));
    cudaMalloc(&devicePixelArrayB, totalThreads * sizeof(*devicePixelArrayB));
    cudaMalloc(&devicePixelArrayDelta, totalThreads * sizeof(*devicePixelArrayDelta));

    for (int colorPlane = 0; colorPlane < depth; colorPlane++) {
        cudaMemcpy(devicePixelArrayA, pixelArraysA[colorPlane], totalThreads * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(devicePixelArrayB, pixelArraysB[colorPlane], totalThreads * sizeof(double), cudaMemcpyHostToDevice);

        // compute squared differences
        covarianceKernel<<<numBlocks, blockSize>>>(devicePixelArrayA, devicePixelArrayB, devicePixelArrayDelta, meanA,
                                                   meanB, totalThreads);
        cudaDeviceSynchronize();

        // a delta array of length totalThreads is now in memory, reduce it so that the first entry is the sum...
        // increase active as a power of 2 until it is larger than totalThreads
        int activeElements = 1;
        while (activeElements <= totalThreads) {
            activeElements = activeElements << 1;
        }

        // sum reduce the elements
        while (activeElements > 1) {
            int reduceThreads = activeElements;
            int reduceBlockSize = THREADS_PER_BLOCK;
            int reduceNumBlocks = (reduceThreads + reduceBlockSize - 1) / reduceBlockSize;
            sumReduce<<<reduceNumBlocks, reduceBlockSize>>>(devicePixelArrayDelta, activeElements, totalThreads);
            cudaDeviceSynchronize();
            activeElements = activeElements >> 1;
        }

        // copy accumulated sum back
        double accumulator;
        cudaMemcpy(&accumulator, devicePixelArrayDelta, 1 * sizeof(double), cudaMemcpyDeviceToHost);

        // add MSE to the result
        result += accumulator;
    }

    // free the arrays
    cudaFree(devicePixelArrayA);
    cudaFree(devicePixelArrayB);
    cudaFree(devicePixelArrayDelta);

    result = result / (width * height * depth);
    return result;
}

// computes structure similarity index measure
double imageOperations::computeSSIM(double **pixelArraysA, double **pixelArraysB, int width, int height, int depth) {
    double meanA = imageOperations::computeMean(pixelArraysA, width, height, depth);
    double meanB = imageOperations::computeMean(pixelArraysB, width, height, depth);
    double varA = imageOperations::computeVariance(pixelArraysA, meanA, width, height, depth);
    double varB = imageOperations::computeVariance(pixelArraysB, meanB, width, height, depth);

    double covariance = imageOperations::computeCovariance(pixelArraysA, pixelArraysB, meanA, meanA, width, height,
                                                           depth);

    double upper = (2 * meanA * meanB + REGULARIZATION_C1) * (2 * covariance + REGULARIZATION_C2);
    double lower = ((meanA * meanA) + (meanB * meanB) + REGULARIZATION_C1) * (varA + varB + REGULARIZATION_C2);

    double result = upper / lower;
    return result;
}