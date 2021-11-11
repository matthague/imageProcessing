#ifndef IMAGEPROCESSING_IMAGEVARIATIONALMETHODS_CUH
#define IMAGEPROCESSING_IMAGEVARIATIONALMETHODS_CUH


namespace imageVariationalMethods {
    extern void
    diffusion(double **pixelArrays, int width, int height, int depth, double deltaTime, double lambda, int numSteps);

    extern void
    total(double **pixelArrays, int width, int height, int depth, double deltaTime, double lambda, int numSteps);
}


#endif //IMAGEPROCESSING_IMAGEVARIATIONALMETHODS_CUH
