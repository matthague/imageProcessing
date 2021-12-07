import sys
import numpy as np
from PIL import Image


def compressPixelData(inputfile, output):
    with open(inputfile, "r") as input:
        data = input.read().replace("\n", " ")
        data = data.replace("  ", " ")
        data = data.split(" ")
        data = data[0:len(data) - 1]
        width = int(data[0], 10)
        height = int(data[1], 10)
        depth = int(data[2], 10)
        data = data[3:]
        for i in range(len(data)):
            data[i] = int(data[i], 10)
        image = np.zeros((height, width, depth), dtype=np.uint8)
        for j in range(height):
            for i in range(width):
                image[j, i] = data[depth * (j * width + i): depth * (j * width + i + 1)]
        if (depth == 4):
            im = Image.fromarray(image, 'RGBA')
        else:
            im = Image.fromarray(image, 'RGB')
        im.save(output)


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("Usage: python3 pixelCompressor  <input_file_name.out> <output_file_name.png>")
        exit()
    compressPixelData(sys.argv[1], sys.argv[2])
