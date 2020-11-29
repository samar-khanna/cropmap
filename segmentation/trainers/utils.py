import os
import torch

from segmentation import MODELS, LOADERS
from data_loaders.dataset import CropDataset


def create_dirs(*dirs):
    """
    Creates directories based on paths passed in as arguments.
    """

    def f_mkdir(p):
        if not os.path.isdir(p):
            print(f"Creating directory {p}")
            os.makedirs(p)

    for p in dirs:
        f_mkdir(p)


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


def create_dataset(classifier_name, *args, **kwargs) -> CropDataset:
    """
    Creates a new initialised CropDataset based on the type of classifier.
    """
    assert classifier_name in LOADERS, \
        "Please specify a valid segmenation classifier available in MODELS"

    dataset_type = LOADERS[classifier_name]

    return dataset_type(*args, **kwargs)
