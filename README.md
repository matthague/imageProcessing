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

# Demo usage and examples

If you need to convert an image to .pxa format, you can do `python3 pixelExtractor filename.format` to get a `filename.pxa` file. I recommend using smaller images (up to 400x400), as some methods have complexity O(n^3).

To just add additive white gaussian noise w/ std. deviation 20
```./imageProcessing filename.pxa filename.out 1 0 20```

![awgn20](https://user-images.githubusercontent.com/33411204/144759611-1313627b-6958-47b9-8145-b8755b527482.png)

To just add salt and pepper noise w/ rate 30
```./imageProcessing filename.pxa filename.out 2 0 30```

![sandp](https://user-images.githubusercontent.com/33411204/144759636-d57a49bf-2012-4114-a79e-cd6c940d7689.png)

To perform total variational reduction on an image with additive white gaussian noise w/ std. deviation 20 using lambda .1
```./imageProcessing filename.pxa filename.out 1 6 20 .1```

![tvawgn](https://user-images.githubusercontent.com/33411204/144759625-bc4fb64a-d983-46f4-b606-4283b9097332.png)

To perform a median filter on an image with salt and pepper noise w/ rate 30 using lambda 1
```./imageProcessing filename.pxa filename.out 2 4 30 1```

![median](https://user-images.githubusercontent.com/33411204/144759646-2ee40db5-6085-4739-96e9-dc5c5c98b79f.png)

To view the processed images you can use `python3 pixelCompressor filename.out filename.png` to get a finished image.
