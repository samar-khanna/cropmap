from models.fcn import FCN
from models.unet import UNet
from models.runet import RUNet
from models.ssavf import SSAVF
from models.m2unet import M2UNet
from models.dumb_net import DumbNet
from models.block_unet import BlockUNet
from models.simple_net import SimpleNet
from models.transformer import Transformer
from models.transformer_feature_extract import TransformerFeatureExtract
from models.knn import kNN

from data_loaders.image_loader import ImageDataset
from data_loaders.time_series_loader import TimeSeriesDataset

from trainers.knn_trainer import KNNTrainer
from trainers.default_trainer import DefaultTrainer
from trainers.maml_trainer import MAMLTrainer
from trainers.ssavf_trainer import SSAVFTrainer
from trainers.simclr_trainer import SimCLRTrainer
from trainers.month_pred_trainer import MonthPredTrainer
from trainers.missing_month_trainer import MissingMonthTrainer


MODELS = {
    "fcn": FCN,
    "unet": UNet,
    "runet": RUNet,
    "m2unet": M2UNet,
    "blockunet": BlockUNet,
    "simplenet": SimpleNet,
    "dumbnet": DumbNet,
    "transformer": Transformer,
    "transformer_feature_extract": TransformerFeatureExtract,
    # "knn": kNN,
    "ssavf": SSAVF,
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
    "transformer_feature_extract": TransformerFeatureExtract,
    "knn": TimeSeriesDataset,
    "dtwnet": TimeSeriesDataset,
    "ssavf": TimeSeriesDataset,
}


TRAINER_TYPES = {
    "default": DefaultTrainer,
    "maml": MAMLTrainer,
    "simclr": SimCLRTrainer,
    "month_pred": MonthPredTrainer,
    "knn": KNNTrainer,
    "missing_month": MissingMonthTrainer,
    "ssavf": SSAVFTrainer,
}
