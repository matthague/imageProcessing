import os
import sys
from PIL import Image


def extractPixelData(inputfile):
    im = Image.open(inputfile)
    pixels = im.load()
    width, height = im.size
    depth = len(pixels[0, 0])
    outputfile = os.path.splitext(inputfile)[0] + '.pxa'
    with open(outputfile, "w", newline="") as output:
        output.write(str(width) + "\n")
        output.write(str(height) + "\n")
        output.write(str(depth) + "\n")
        for j in range(height):
            for i in range(width):
                currentPixel = pixels[i, j]
                for k in range(depth):
                    output.write(str(currentPixel[k]) + " ")
                if (i == width - 1):
                    output.write("\n")


if __name__ == "__main__":
    extractPixelData(sys.argv[1])
