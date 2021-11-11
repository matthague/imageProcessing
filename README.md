# imageProcessing | Survey of Image Denoising Techniques

Howdy! If you're reading this it means that you've stumbled across a bit of code that I've written as a class project.

The code in this repo is designed to perform a variety of image denoising techniques in CUDA. The code should be built using CMAKE,
but a manual makefile is also included for those that don't mind specifying include paths manually.

The primary built target is called imageProcessing, and it contains a few methods that
* add additive white gaussian noise
* add salt and pepper noise
* do a median filter
* do svd on an image
* do a fourier low pass filter
* do variational diffusion
* do total variation minimization
* and more!

Python scripts are included to extract(pixelExtractor) images into a .pxa (pixel array) format, and compress(pixelCompressor) .pxa files back into .png or other image formats.

Good luck. Hopefully this code is ~helpful~ *useful* to you.
