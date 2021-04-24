from models.fcn import FCN
from models.unet import UNet
from models.runet import RUNet
from models.m2unet import M2UNet
from models.block_unet import BlockUNet
from models.simple_net import SimpleNet
from models.dumb_net import DumbNet
from models.transformer import Transformer
from models.dtwnet import DTWNET_SINGLE
from models.knn import kNN

from data_loaders.image_loader import ImageDataset
from data_loaders.time_series_loader import TimeSeriesDataset


MODELS = {
    "fcn": FCN,
    "unet": UNet,
    "runet": RUNet,
    "m2unet": M2UNet,
    "blockunet": BlockUNet,
    "simplenet": SimpleNet,
    "dumbnet": DumbNet,
    "transformer": Transformer,
    "knn": kNN,
    "dtwnet": DTWNET_SINGLE
}


LOADERS = {
    "fcn": ImageDataset,
    "unet": ImageDataset,
    "runet": TimeSeriesDataset,
    "m2unet": TimeSeriesDataset,
    "blockunet": TimeSeriesDataset,
    "simplenet": TimeSeriesDataset,
    "dumbnet": TimeSeriesDataset,
    "transformer": TimeSeriesDataset,
    "knn": TimeSeriesDataset,
    "dtwnet": TimeSeriesDataset
}
