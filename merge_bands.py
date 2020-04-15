from concurrent.futures import ProcessPoolExecutor
from glob import glob
import os
from os.path import join
from pprint import pprint #TODO delete this
import sys

import rasterio

SR_BANDS = map(lambda n: 'SRB{}'.format(n), [1,2,3,4,5,6,7])
BT_BANDS = map(lambda n: 'BTB{}'.format(n), [10,11])
ST_BAND = ['ST']
BANDS = [it for l in [SR_BANDS, BT_BANDS, ST_BAND] for it in l]

def get_band_files(data_root, folders):
    def filename(folder, band):
        return '{}_{}.tif'.format(folder, band)

    def get_folder_bands(folder):
        return map(lambda b: join(data_root, folder, filename(folder, b)), BANDS)

    return {f: get_folder_bands(f) for f in folders}


def merge_bands(data_root, entityId, band_files):
    # TODO clean up
    dataset_objs = list(map(lambda f: rasterio.open(f), band_files))
    raster_objs = map(lambda d: d.read(1), dataset_objs)
    comp_fname = join(data_root, entityId+'_'+'composite.tif')
    composite = rasterio.open(
        comp_fname,
        'w',
        driver='GTiff',
        height=dataset_objs[0].height,
        width=dataset_objs[0].width,
        count=len(dataset_objs),
        dtype=dataset_objs[0].dtypes[0],
        crs=dataset_objs[0].crs,
        nodata=dataset_objs[0].nodata,
        transform=dataset_objs[0].transform)
    for i,r,d in zip(range(len(dataset_objs)), raster_objs, dataset_objs):
        composite.write(r, i+1)
        d.close()

    composite.close()


def main(data_root):
    folders = os.listdir(data_root)
    folder_files = get_band_files(data_root, folders)
    #pprint({k: list(m) for k,m in folder_files.items()})
    packed = list(folder_files.items())
    for entityId, band_files in folder_files.items():
        merge_bands(data_root, entityId, band_files)

if __name__ == '__main__':
    data_root = sys.argv[1]
    main(data_root)
