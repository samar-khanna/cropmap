import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from models.fcn import FCN
from models.unet import UNet
from models.m2unet import M2UNet
from models.block_unet import BlockUNet
from models.simple_net import SimpleNet
import models.loss as custom_loss

from data_loaders.dataset import CropDataset
from data_loaders.image_loader import ImageDataset
from data_loaders.time_series_loader import TimeSeriesDataset


MODELS = {
    "fcn": FCN,
    "unet": UNet,
    "m2unet": M2UNet,
    "blockunet": BlockUNet,
    "simplenet": SimpleNet
}


LOADERS = {
    "fcn": ImageDataset,
    "unet": ImageDataset,
    "m2unet": TimeSeriesDataset,
    "blockunet": TimeSeriesDataset,
    "simplenet": TimeSeriesDataset
}


class ConfigHandler():
    def __init__(self, data_path, path_to_config, classes_path, out_dir,
                 inf_dir=None, inf_subdir=None):
        super().__init__()

        self.data_path = data_path

        # Load classes.
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        self.classes = classes

        with open(path_to_config, 'r') as f:
            config = json.load(f)

        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

        self.num_classes = len(self.config.get("interest_classes") or self.classes.keys())

        # TODO: Maybe package this functionality outside of this class to avoid confusion
        # Create out directories.
        self.out_dir = os.path.join(out_dir, self.name)
        self.save_dir = os.path.join(self.out_dir, "checkpoints")
        self.metrics_dir = os.path.join(self.out_dir, "metrics")
        self.inf_dir = inf_dir if inf_dir is not None else \
            os.path.join(self.out_dir, "inference")
        if inf_subdir:
            self.inf_dir = os.path.join(self.inf_dir, inf_subdir)
        ConfigHandler._create_dirs(self.out_dir, self.save_dir,
                                   self.metrics_dir, self.inf_dir)

        self.save_path = os.path.join(self.save_dir, f"{self.name}.bin")

    def save_config(self):
        """
        Saves the config file as a .json file in `self.out_dir`.
        """
        config_path = os.path.join(self.out_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved model config file at: {config_path}")

    @staticmethod
    def _create_dirs(*dirs):
        """
        Creates directories based on paths passed in as arguments.
        """
        def f_mkdir(p):
            if not os.path.isdir(p):
                print(f"Creating directory {p}")
                os.makedirs(p)

        for p in dirs: f_mkdir(p)


def create_model(model_config, num_classes):
    """
    Creates a new segmentation model given the config dictionary.
    Uses the specialised creator functions for each model.
    """
    assert model_config["classifier"].lower() in MODELS, \
        "Please specify a valid segmenation classifier available in MODELS"

    # Create Segmentation model
    seg_model_class = MODELS[model_config["classifier"].lower()]
    seg_model = seg_model_class.create(model_config, num_classes)

    return seg_model


def load_model(model_config, num_classes, from_checkpoint=None, freeze_backbone=False):
    """
    Loads a segmentation model based on its config dictionary.
    If specified, load's model weights from a checkpoint file.
    Else, creates a new, fresh instance of the model.
    If specified, also freezes all parameters in backbone layer.
    """
    model = create_model(model_config, num_classes)

    if from_checkpoint:
        if type(from_checkpoint) is str:
            checkpoint_path = from_checkpoint
            assert os.path.isfile(from_checkpoint), \
                f"Model's .bin checkpoint file doesn't exist at: {checkpoint_path}"
        else:
            raise ValueError(f"Keyword arg `from_checkpoint` must be a string")

        print(f"Loading model weights from checkpoint path {checkpoint_path}")

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    # TODO: Finetuning (freezing layers other than backbone)
    # Freeze backbone if specified
    if freeze_backbone:
        print("Freezing backbone layers...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def save_model(model, save_path):
    """
    Saves a segmentation model to `save_path`.
    If using multiple gpus/data parallelism, then save `.module` attribute.
    """
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"Saved model weights at: {save_path}")


def get_loss_optimizer(config, model, device):
    """
    Instantiates loss function and optimizer based on name and kwargs.
    Ensure that names are valid in the torch.nn and torch.optim modules.
    Also ensure keyword arguments match.
    Defaults to using BinaryCrossentropy (from logits), and Adam(lr=0.0001)
    """
    # Set up loss.
    loss_name = config.get("loss", "BCEWithLogitsLoss")
    loss_kwargs = config.get("loss_kwargs", {})

    # If weights are given, then convert to tensor and normalize to sum to 1.
    for key in loss_kwargs:
        if key.find("weight") > -1:
            loss_kwargs[key] = torch.as_tensor(
                loss_kwargs[key], dtype=torch.float32, device=device
            ) / sum(loss_kwargs[key])
    if loss_name in custom_loss.__dict__:
        loss_fn = custom_loss.__dict__[loss_name](**loss_kwargs)
    elif loss_name in nn.__dict__:
        loss_fn = nn.__dict__[loss_name](**loss_kwargs)
    else:
        raise ValueError(("Invalid PyTorch loss. The name must exactly match a loss" 
                          " in the nn module or a loss defined in models/loss.py"))

    # Set up optimizer
    optim_name = config.get("optimizer", "Adam")
    optim_kwargs = config.get("optimizer_kwargs", {"lr": 0.001})
    assert optim_name in optim.__dict__, \
        "Invalid PyTorch optimizer. The name must exactly match an optimizer in the optim module"

    # Only optimize on unfrozen weights.
    weights = filter(lambda w: w.requires_grad, model.parameters())
    optimizer = optim.__dict__[optim_name](weights, **optim_kwargs)
    return loss_fn, optimizer


def create_dataset(classifier_name, *args, **kwargs) -> CropDataset:
    """
    Creates a new initialised CropDataset based on the type of classifier.
    """
    assert classifier_name in LOADERS, \
        "Please specify a valid segmenation classifier available in MODELS"

    dataset_type = LOADERS[classifier_name]

    return dataset_type(*args, **kwargs)
