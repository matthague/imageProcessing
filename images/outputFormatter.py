import numpy as np
import sys
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
    strs = "barnawgn.out barndiffuseawgn.out barndiffusesandp.out barnfourierawgn.out barnfouriersandp.out barnmedianawgn.out barnmediansandp.out barnsandp.out barnsvdawgn.out barnsvdsandp.out barntotalawgn.out barntotalsandp.out butterflyawgn.out butterflydiffuseawgn.out butterflydiffusesandp.out butterflyfourierawgn.out butterflyfouriersandp.out butterflymedianawgn.out butterflymediansandp.out butterflysandp.out butterflysvdawgn.out butterflysvdsandp.out butterflytotalawgn.out butterflytotalsandp.out cameraawgn.out cameradiffuseawgn.out cameradiffusesandp.out camerafourierawgn.out camerafouriersandp.out cameramedianawgn.out cameramediansandp.out camerasandp.out camerasvdawgn.out camerasvdsandp.out cameratotalawgn.out cameratotalsandp.out moonawgn.out moondiffuseawgn.out moondiffusesandp.out moonfourierawgn.out moonfouriersandp.out moonmedianawgn.out moonmediansandp.out moonsandp.out moonsvdawgn.out moonsvdsandp.out moontotalawgn.out moontotalsandp.out"
    infiles = strs.split(" ")
    outfiles = []
    for a in infiles:
        b = a.split(".")[0] + ".png"
        outfiles.append(b)

    assert(len(infiles) == len(outfiles))
    for i in range(len(infiles)):
        compressPixelData(infiles[i], outfiles[i])
