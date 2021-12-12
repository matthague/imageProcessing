# imageProcessing | Survey of Image Denoising Techniques

Howdy! If you're reading this it means that you've stumbled across a bit of code that I've written as a class project.

The code in this repo is designed to perform a variety of image denoising techniques in CUDA. This code is intended to add noise to clean base images, and then remove it using some of the included methods--however, it can be used on images without adding noise by setting the <noise_rate> paramater to 0. 

The code should be built using cmake, but a manual makefile is also included for those on Vocareum.

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

# Building the code
cmake version 3.21 or later is required to build the code for this project. If you are in the Vocareum enviroment, a MANUALMakefile has been provided, and should be used. To install the required Python 3 packages, and compile the main code with cmake run the following:
```
cd /.../.../imageProcessing/
pip install -r requirements.txt # may need to be pip3 
mkdir build
cd build
cmake ..
make -j
```
Ta da! Hopefully everything built correctly. 

If you are on Vocareum, running a simple:
```
cd /.../.../imageProcessing/
pip install -r requirements.txt # may need to be pip3 (NOTE: vocareum should already have the python packages installed, should be okay if this fails)
mv MANUALMakefile Makefile
make all
```
Should do the trick.

# Demo usage and examples

I recommend using only smaller images (up to 400x400), as it can be tricky to see the effects with lots and lots of pixels.

**If you do not have a testing image--images/camera.png, images/butterfly.png, images/moon.png, or images/barn.png--can be used by copying them to your working/build directory.**

If you need to convert an image to .pxa format, you can do `python3 pixelExtractor.py inputfilename.png` to get a `inputfilename.pxa` file. Note: Most file formats are supported via the Python Image Library (PIL), so you can use .jpeg, .png, etc.

To just add additive white gaussian noise w/ std. deviation 20
```./imageProcessing inputfilename.pxa outputfilename.out 1 0 20```

![awgn20](https://user-images.githubusercontent.com/33411204/144759611-1313627b-6958-47b9-8145-b8755b527482.png)

To just add salt and pepper noise w/ rate 30
```./imageProcessing inputfilename.pxa outputfilename.out 2 0 30```

![sandp](https://user-images.githubusercontent.com/33411204/144759636-d57a49bf-2012-4114-a79e-cd6c940d7689.png)

To perform total variational reduction on an image with additive white gaussian noise w/ std. deviation 20 using lambda .1
```./imageProcessing inputfilename.pxa outputfilename.out 1 6 20 .1```

![tvawgn](https://user-images.githubusercontent.com/33411204/144759625-bc4fb64a-d983-46f4-b606-4283b9097332.png)

To perform a median filter on an image with salt and pepper noise w/ rate 30 using lambda 1
```./imageProcessing inputfilename.pxa outputfilename.out 2 4 30 1```

![median](https://user-images.githubusercontent.com/33411204/144759646-2ee40db5-6085-4739-96e9-dc5c5c98b79f.png)

To view the processed images you can use `python3 pixelCompressor.py outputfilename.out outputfilename.png` to get a finished image.
