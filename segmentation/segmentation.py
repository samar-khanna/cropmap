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


class ConfigHandler():
  def __init__(self, data_path, path_to_config, classes_path, 
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

    # Filter out interested classes, if specified. Otherwise use all classes.
    # Sort the classes in order of their indices
    interest_classes = self.config.get("interest_classes")
    interest_classes = interest_classes if interest_classes else classes.keys()
    interest_classes = sorted(interest_classes, key=self.classes.get)

    # orig_to_new_ind maps from original class index --> new class index
    # New classes is the mapping of class --> new_ind
    orig_to_new_ind = {}
    new_classes = {}
    for i, class_name in enumerate(interest_classes):
      orig_to_new_ind[self.classes[class_name]] = i
      new_classes[class_name] = i
    self.classes = new_classes
    self.index_map = orig_to_new_ind
    
    # Indices according to train/val/test split will be stored here.    
    self.indices_path = os.path.join(data_path, "indices.json")

    # Create out directories.
    self.out_dir = os.path.join(config.get("out_dir", "."), self.name)
    self.save_dir = os.path.join(self.out_dir, "checkpoints")
    self.metrics_dir = os.path.join(self.out_dir, "metrics")
    self.inf_dir = inf_dir if inf_dir else \
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
    
    print(f"Loading model weights from checkpoint path {checkpoint_path}")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

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
