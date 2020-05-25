import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.models.resnet as resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from fcn import FCN
from unet import UNet


MODELS = {
  "fcn": FCN,
  "unet": UNet
}


def create_model(config_handler):
  """
  Creates a new segmentation model given the config dictionary.
  Initialises a ResNet backbone (optionally pretrained) for the segmentation model.
  """
  config = config_handler.config
  backbone_name = config["backbone"].lower()
  assert backbone_name.find("resnet") > -1, "Only resnet backbones supported"

  # Initialise ResNet backbone
  backbone = resnet.__dict__[backbone_name](
    # replace_stride_with_dilation=[False, True, True],
    **config.get("backbone_kwargs", {})
  )

  # Reformat first conv to take in new #channels
  in_channels = config.get("input_shape", [3])[0]
  old_conv = backbone.conv1
  backbone.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding)
  if config.get("backbone_kwargs", {}).get("pretrained"):
    backbone.conv1.weight[:, :3] = old_conv.weight

  assert config["classifier"].lower() in MODELS,\
    "Please specify a valid segmenation classifier available in MODELS"

  # Create Segmentation model
  num_classes = len(config_handler.classes)
  seg_model = MODELS[config["classifier"].lower()]
  seg_model = seg_model(backbone, num_classes=num_classes, **config["classifier_kwargs"])

  return seg_model
  

def load_model(config_handler, from_checkpoint=False):
  """
  Loads a segmentation model based on its config dictionary.
  If specified, load's model weights from a checkpoint file.
  Else, creates a new, fresh instance of the model.
  """
  model = create_model(config_handler)

  if from_checkpoint:
    if type(from_checkpoint) is str:
      checkpoint_path = from_checkpoint
      assert os.path.isfile(from_checkpoint), \
        f"Model's .bin checkpoint file doesn't exist at: {checkpoint_path}"
    elif type(from_checkpoint) is bool:
      checkpoint_path = config_handler.save_path
      assert os.path.isfile(config_handler.save_path), \
        f"Model's .bin checkpoint file doesn't exist on config path: {checkpoint_path}"
    else:
      raise ValueError(f"Keyword arg `from_checkpoint` must be either a bool or str")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

  return model


def save_model(model, config_handler):
  """
  Saves a segmentation model either to the directory specified in `save_dir`,
  or to the path specified in the model config.
  Also saves the model config as a json file in that directory.
  """
  save_dir = config_handler.save_dir
  save_path = config_handler.save_path

  torch.save(model.state_dict(), save_path)
  with open(os.path.join(save_dir, "config.json"), 'w') as f:
    json.dump(config_handler.config, f)
