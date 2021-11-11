#ifndef IMAGEPROCESSING_IMAGEOPERATIONS_CUH
#define IMAGEPROCESSING_IMAGEOPERATIONS_CUH

#include <curand.h>
#include <curand_kernel.h>

namespace imageOperations {
    extern void
    addGaussianNoise(double **pixelArrays, curandState *rngState, int width, int height, int depth, double noiseStdev);

    extern void
    addSANDPNoise(double **pixelArrays, curandState *rngState, int width, int height, int depth, double noiseRate);

    extern double computeMSE(double **pixelArraysA, double **pixelArraysB, int width, int height, int depth);

    extern double computePSNR(double mse);

    extern double computeMean(double **pixelArrays, int width, int height, int depth);

    extern double computeVariance(double **pixelArrays, double mean, int width, int height, int depth);

    extern double
    computeCovariance(double **pixelArraysA, double **pixelArraysB, double meanA, double meanB, int width, int height,
                      int depth);

    extern double computeSSIM(double **pixelArraysA, double **pixelArraysB, int width, int height, int depth);

    extern void initRNG(curandState *state, int stateSize, int seed);
}

#endif //IMAGEPROCESSING_IMAGEOPERATIONS_CUH
