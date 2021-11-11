#ifndef IMAGEPROCESSING_IMAGEHANDLING_CUH
#define IMAGEPROCESSING_IMAGEHANDLING_CUH

#include <stdio.h>

namespace imageHandling {
    extern void getImageDimensions(FILE *file, int *width, int *height, int *depth);

    extern void loadImage(FILE *file, double **pixelArrays, int width, int height, int depth);

    extern void saveImage(FILE *outputFile, double **pixelArrays, int width, int height, int depth);

    extern double **allocateImage(int depth, int width, int height);

    extern void freeImage(double **pixelArrays, int depth);
}

#endif //IMAGEPROCESSING_IMAGEHANDLING_CUH
