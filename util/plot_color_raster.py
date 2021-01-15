import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import numpy as np
import rasterio

from sys import argv, exit

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))


def display(mosaic_path):
    rgb_bands = [4,3,2]
    with rasterio.open(mosaic_path, 'r') as mosaic:
        r,g,b = map(lambda b: normalize(mosaic.read(b)), rgb_bands)
        rgb = np.dstack([r,g,b])
        pyplot.imshow(rgb)
        pyplot.show()


if __name__ == '__main__':
    if len(argv) < 2:
        print('Need to provide a path to mosaic')
        exit(-1)

    display(argv[1])
