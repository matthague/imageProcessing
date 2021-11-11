#ifndef IMAGEPROCESSING_IMAGEFILTERS_CUH
#define IMAGEPROCESSING_IMAGEFILTERS_CUH

namespace imageFilters {
    extern void fourierLowPass(double **pixelArrays, int width, int height, int depth, float cutoff);

    extern void median(double **pixelArrays, int width, int height, int depth, int passes);
}

#endif //IMAGEPROCESSING_IMAGEFILTERS_CUH
