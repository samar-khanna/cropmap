import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import numpy as np
import rasterio

from argparse import ArgumentParser
from sys import argv, exit
from traceback import print_exc



def display(mosaic_path, scale=1.0):
    def _normalize(array):
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))

    def _norm_read(mosaic, band, h, w):
        return _normalize(mosaic.read(band, out_shape=(1, h, w)))

    rgb_bands = [4,3,2]
    with rasterio.open(mosaic_path, 'r') as m:
        h, w = m.shape
        h, w = int(scale*h), int(scale*w)
        r,g,b = map(lambda b: _norm_read(m, b, h, w), rgb_bands)
        rgb = np.dstack([r,g,b])
        pyplot.imshow(rgb)
        pyplot.show()



if __name__ == '__main__':
    p = ArgumentParser(description='Display a landsat8 mosaic, in whole or part')

    p.add_argument('mosaic_path', help='path to mosaic file')

    p.add_argument('--scale', '-s', type=float, default=1.0)
    args = p.parse_args()


    try:
        display(args.mosaic_path, args.scale)
    except Exception as E:
        print(E)
        print_exc()
        exit(-1)
