#include "imageHandling.cuh"

#include <iostream>
#include <stdio.h>

void imageHandling::getImageDimensions(FILE *file, int *width, int *height, int *depth) {
    fscanf(file, "%d", width);
    fscanf(file, "%d", height);
    fscanf(file, "%d", depth);
}

void imageHandling::loadImage(FILE *file, double **pixelArrays, int width, int height, int depth) {
    int offset = 0;
    while (!feof(file)) {
        for (int i = 0; i < depth; i++) {
            double tval;
            fscanf(file, "%lf", &tval);
            pixelArrays[i][offset] = tval;
        }
        offset += 1;
    }

    fclose(file);
}

void imageHandling::saveImage(FILE *outputFile, double **pixelArrays, int width, int height, int depth) {
    fprintf(outputFile, "%d\n", width);
    fprintf(outputFile, "%d\n", height);
    fprintf(outputFile, "%d\n", depth);

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            for (int k = 0; k < depth; k++) {
                int tval = (int) round(pixelArrays[k][j * width + i]);
                // clamp the values
                if (tval < 0) {
                    tval = 0;
                } else if (tval > 255) {
                    tval = 255;
                }
                fprintf(outputFile, "%d ", tval);
            }
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);
}

double **imageHandling::allocateImage(int depth, int width, int height) {
    double **pixelArrays;
    cudaMallocHost(&pixelArrays, depth * sizeof(*pixelArrays));

    for (int i = 0; i < depth; i++) {
        cudaMallocHost(&(pixelArrays[i]), width * height * sizeof(double));
    }

    return pixelArrays;
}

void imageHandling::freeImage(double **pixelArrays, int depth) {
    for (int i = 0; i < depth; i++) {
        cudaFreeHost(pixelArrays[i]);
    }

    cudaFreeHost(pixelArrays);
}