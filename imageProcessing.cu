/*
    LICENCE NOTICE
    imageProcessing | A program for image denoising experiments.
    Copyright (C) 2021  Matt Hague

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <iostream>
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#include "imageHandling.cuh"
#include "imageOperations.cuh"
#include "imageAlgebra.cuh"
#include "imageFilters.cuh"
#include "imageVariationalMethods.cuh"

#define GAUSSIAN_NOISE_MODE 1
#define SANDP_NOISE_MODE 2

#define MIN_PROCESSING_MODE 0
#define GRAYSCALE_MODE 1
#define SVD_MODE 2
#define FOURIER_LOW_PASS_MODE 3
#define MEDIAN_MODE 4
#define DIFFUSION_MODE 5
#define TOTAL_VARIATION_MODE 6
#define MAX_PROCESSING_MODE 6

#define DEFAULT_NUM_ITERATIONS 1 // SET THIS TO WHATEVER YOU LIKE
#define NO_ALPHA true // SET THIS FALSE IF YOU WANT TO PROCESS AN ALPHA CHANNEL

void printUsage(char *programName) {
    printf("Usage:\n");
    printf("\t%s <input_file> <output_file> <noise_mode> <processing_mode> <noise_rate> <OPTION:lambda1> <OPTION:num_iterations>\n\n",
           programName);
}

void printModes() {
    printf("Noise modes:\n");
    printf("\t1: Additive white gaussian - has standard deviation of <noise_rate>.\n");
    printf("\t2: Salt and pepper - has frequency of <noise_rate>/256.\n\n");

    printf("Processing modes:\n");
    printf("\t0: NULL - Does nothing.\n");
    printf("\t1: Grayscale - An intermediate mode for debugging. No options.\n");
    printf("\t2: SVD - Performs a singular value decomposition approximation, keeping the first <lambda1> fraction of singular values.\n");
    printf("\t3: Fourier Low Pass - Performs a radial low pass filter on the 2d transform of the image, using parameter <lambda1> for radius. Increasing the radius means that the image will be represented by more frequencies.\n");
    printf("\t4: Median - Performs a 3x3 sliding window median filter on the image, using parameter <lambda1> for the number of filter passes. <lambda1> should be a positive integer.\n");
    printf("\t5: Diffusion - Performs a variational diffusion on the image, using parameter <lambda1> for the diffusive regularization parameter. <lambda1> should be a positive float.\n");
    printf("\t6: Total Variation - Performs a total variation minimization on the image, using parameter <lambda1> for the regularization parameter. <lambda1> should be a positive float.\n\n");
}

int parseArgs(int argc, char *argv[], FILE **inputFile, FILE **outputFile, int *noiseMode, int *processingMode,
              double *noiseRate,
              double *lambda1, int* num_iters) {
    // basic parsing
    if (argc < 6) {
        printf("Error: invalid number of arguments.\n\n");
        printUsage(argv[0]);
        printModes();
        return 1;
    }

    *inputFile = fopen(argv[1], "r");
    if (*inputFile == NULL) {
        printf("Error: could not open %s\n\n", argv[1]);
        perror("FATAL");
        return 1;
    }

    *outputFile = fopen(argv[2], "w");
    if (*outputFile == NULL) {
        printf("Error: could not open %s\n\n", argv[2]);
        perror("FATAL");
        return 1;
    }

    *noiseMode = atoi(argv[3]);
    if (*noiseMode != GAUSSIAN_NOISE_MODE && *noiseMode != SANDP_NOISE_MODE) {
        printf("Error: Invalid noise mode\n\n");
        printUsage(argv[0]);
        printModes();
        return 1;
    }

    *processingMode = atoi(argv[4]);
    if (*processingMode < MIN_PROCESSING_MODE || *processingMode > MAX_PROCESSING_MODE) {
        printf("Error: Invalid processing mode\n\n");
        printUsage(argv[0]);
        printModes();
        return 1;
    }

    *noiseRate = atof(argv[5]);
    if (*noiseRate < 0.0 || *noiseRate > 255.0) {
        printf("Error: <noise_rate> must be in range [0.0 ... 255.0]\n\n");
        printUsage(argv[0]);
        return 1;
    }

    // process optional arguments
    if (*processingMode == SVD_MODE || *processingMode == FOURIER_LOW_PASS_MODE) {
        if (argc < 7) {
            printf("Error: must specify <lambda1>\n\n");
            printUsage(argv[0]);
            return 1;
        }
        *lambda1 = atof(argv[6]);
        if (*lambda1 < 0.0 || *lambda1 > 1.0) {
            printf("Error: <lambda1> must be in range [0.0 ... 1.0]\n\n");
            printUsage(argv[0]);
            return 1;
        }
    }

    if(*processingMode == MEDIAN_MODE) {
        if (argc < 7) {
            printf("Error: must specify <lambda1>\n\n");
            printUsage(argv[0]);
            return 1;
        }
        *lambda1 = atoi(argv[6]);
        if(*lambda1 < 1) {
            printf("Error: <lambda1> must be a positive integer\n\n");
            printUsage(argv[0]);
            return 1;
        }
    }

    if(*processingMode == DIFFUSION_MODE || *processingMode == TOTAL_VARIATION_MODE) {
        if (argc < 7) {
            printf("Error: must specify <lambda1>\n\n");
            printUsage(argv[0]);
            return 1;
        }
        *lambda1 = atof(argv[6]);
        if(*lambda1 < 0.0) {
            printf("Error: <lambda1> must be a positive float\n\n");
            printUsage(argv[0]);
            return 1;
        }
    }

    if(argc >= 8) {
        *num_iters = atoi(argv[7]);
        if(*num_iters < 1) {
            printf("Error: <num_iterations> cannot be less than 1");
            printUsage(argv[0]);
            return 1;
        }
    }

    return 0;
}

void
processImage(double **inputPixelArrays, double **outputPixelArrays, int width, int height, int depth,
             int noiseMode, int processingMode,
             double noiseRate, double lambda1, int numIterations) {
    // image similarity measures
    double averageMSE = 0.0;
    double averageSSIM = 0.0;

    // set up the RNG state
    curandState *rngState;
    cudaMalloc(&rngState, width * height * sizeof(*rngState));
    imageOperations::initRNG(rngState, width * height, time(0));

    for (int iteration = 0; iteration < numIterations; iteration++) {
        // copy true input pixel data to the output array
        for (int i = 0; i < depth; i++) {
            cudaMemcpy(outputPixelArrays[i], inputPixelArrays[i], width * height * sizeof(double),
                       cudaMemcpyHostToHost);
        }

        // add artificial noise to the image
        switch (noiseMode) {
            case GAUSSIAN_NOISE_MODE:
                imageOperations::addGaussianNoise(outputPixelArrays, rngState, width, height, depth, noiseRate);
                break;

            case SANDP_NOISE_MODE:
                imageOperations::addSANDPNoise(outputPixelArrays, rngState, width, height, depth, noiseRate);
                break;

            default:
                break;
        }

        // additional variables used in some processing modes
        int k;

        // denoise the image
        switch (processingMode) {
            case GRAYSCALE_MODE:
                // basic grayscale conversion
                for (int j = 0; j < height; j++) {
                    for (int i = 0; i < width; i++) {
                        double avg = outputPixelArrays[0][j * width + i] + outputPixelArrays[1][j * width + i] +
                                    outputPixelArrays[2][j * width + i] / 3;
                        outputPixelArrays[0][j * width + i] = avg;
                        outputPixelArrays[1][j * width + i] = avg;
                        outputPixelArrays[2][j * width + i] = avg;
                    }
                }
                break;

            case SVD_MODE:
                k = ceil(min(width, height) * lambda1);
                imageAlgebra::kSVD(outputPixelArrays, width, height, depth, k);
                break;

            case FOURIER_LOW_PASS_MODE:
                imageFilters::fourierLowPass(outputPixelArrays, width, height, depth, lambda1);
                break;

            case MEDIAN_MODE:
                k = (int) lambda1;
                imageFilters::median(outputPixelArrays, width, height, depth, k);
                break;

            case DIFFUSION_MODE:
                imageVariationalMethods::diffusion(outputPixelArrays, width, height, depth, 0.001, lambda1, 250);
                break;

            case TOTAL_VARIATION_MODE:
                imageVariationalMethods::total(outputPixelArrays, width, height, depth, 0.1, lambda1, 250);
                break;

            default:
                break;
        }

        // accumulate statistics to get average
        double mse = imageOperations::computeMSE(inputPixelArrays, outputPixelArrays, width, height, depth);
        double ssim = imageOperations::computeSSIM(inputPixelArrays, outputPixelArrays, width, height, depth);
        averageMSE += mse / numIterations;
        averageSSIM += ssim / numIterations;
    }

    // numerically stable way to get PSNR
    double averagePSNR = imageOperations::computePSNR(averageMSE);

    // display final statistics
    printf("Number of trials: %d\n", numIterations);
    printf("Average MSE: %f\n", averageMSE); // mean squared error
    printf("Average PSNR: %f dB\n", averagePSNR); // peak signal-to-noise ratio
    printf("Average SSIM: %f\n", averageSSIM); // structure similarity index measurement

    // free the RNG state
    cudaFree(rngState);
}

int main(int argc, char *argv[]) {
    // parse arguments
    FILE *inputFile;
    FILE *outputFile;
    int noiseMode;
    int processingMode;
    double noiseRate;
    double lambda1;
    int numIterations = DEFAULT_NUM_ITERATIONS;

    int err = parseArgs(argc, argv, &inputFile, &outputFile, &noiseMode, &processingMode, &noiseRate, &lambda1, &numIterations);
    if (err) {
        return 1;
    }

    // load image
    int width; // image width
    int height; // image height
    int depth; // number of color planes [e.g. 4 for RGBA]
    imageHandling::getImageDimensions(inputFile, &width, &height, &depth);

    double **inputPixelArrays; // pointers to arrays that hold pixel data
    double **outputPixelArrays;
    inputPixelArrays = imageHandling::allocateImage(depth, width, height);
    outputPixelArrays = imageHandling::allocateImage(depth, width, height);
    imageHandling::loadImage(inputFile, inputPixelArrays, width, height, depth);

    // set effective depth
    int effectiveDepth = depth;
    if (NO_ALPHA && (effectiveDepth == 4)) {
        effectiveDepth -= 1;
        cudaMemcpy(outputPixelArrays[depth - 1], inputPixelArrays[depth - 1], width * height * sizeof(double),
                   cudaMemcpyHostToHost);
    }

    // process the image according to user parameters
    processImage(inputPixelArrays, outputPixelArrays, width, height, effectiveDepth, noiseMode, processingMode,
                 noiseRate, lambda1, numIterations);

    // save output and free memory
    imageHandling::saveImage(outputFile, outputPixelArrays, width, height, depth);
    imageHandling::freeImage(inputPixelArrays, depth);
    imageHandling::freeImage(outputPixelArrays, depth);
    cudaDeviceReset();

    return 0;
}
