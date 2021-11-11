cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(imageProcessing CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)

find_package(CUDAToolkit)

add_executable(imageProcessing imageProcessing.cu imageHandling.cu imageHandling.cuh imageOperations.cu imageOperations.cuh imageAlgebra.cu imageAlgebra.cuh imageFilters.cu imageFilters.cuh imageVariationalMethods.cu imageVariationalMethods.cuh)

target_link_libraries(imageProcessing PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)

set_target_properties(
        imageProcessing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)