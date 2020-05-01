import os
import csv
import json
import random
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class ConfigHandler():
  def __init__(self, data_path, path_to_config, classes_path):
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
    self.inf_dir = os.path.join(self.out_dir, "inference")
    ConfigHandler._create_dirs(self.out_dir, self.save_dir, 
                               self.metrics_dir, self.inf_dir)

    self.save_path = os.path.join(self.save_dir, f"{self.name}.bin")

    
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
  

class CropDataset(Dataset):
  def __init__(self, config_handler, tile_size=(224, 224), overlap=0,
               train_val_test=[0.8, 0.1, 0.1], inf_mode=False, transform=None):
    """
    Initialises an instance of a `CropDataset`.
    Requires:
      `config_handler`: Object that handles the model config file.
      `tile_size`: (h,w) denoting size of each tile to sample from area.
      `overlap`: number of pixels adjacent tiles share.
      `train_val_test`: Percentage split (must add to 1) of data sizes
      `inf_mode`: Whether the dataset is being used for inference or not.
                  If not for inference, then make sure that labels exist.
      `transform`: Function to augment data.
    """
    self.transform = transform
    self.tile_size = tile_size
    self.overlap = overlap
    self.train_val_test = train_val_test
    self.inf_mode = inf_mode

    self.num_classes = len(config_handler.classes)
    self.map_index = lambda i: config_handler.index_map.get(i, -1)

    # Mosaic tile
    mosaic_path = os.path.join(config_handler.data_path, 'mosaic.tif')
    self.mosaic = rasterio.open(mosaic_path)

    # Ground truth labels
    mask_path = os.path.join(config_handler.data_path, 'mask.tif')
    self.mask_exists = os.path.isfile(mask_path)
    if self.mask_exists:
      self.mask = rasterio.open(mask_path)

  def __len__(self):
    h, w = self.mosaic.shape
    th, tw = self.tile_size
    o = self.overlap
    total_rows = h//(th - o)
    total_cols = w//(tw - o)
    return self.total_rows * self.total_cols
  
  @staticmethod
  def one_hot_mask(mask, num_classes):
    """
    For a given mask of (1, h, w) dimensions, where each pixel value
    represents the index of the class, this expands the mask into a 
    (c, h, w) mask where each depth slice for index j is a one-hot
    encoding for pixels that belong to class j.
    """
    # Expand y in to (w, h, c) array (for multiplication)
    y = np.ones((mask.shape[2], mask.shape[1], num_classes))

    # y is 3d array of repeated tile indices, ith slice is
    # 2d array of dim (h, w) full of values i
    y = y * np.arange(1, num_classes+1)
    y = y.T

    # One hot encoding by checking for equality with dense mask
    y = y == mask

    return y

  def __getitem__(self, index):
    """
    This returns only ONE sample from the dataset, for a given index.
    The returned sample should be a tuple (x, y) where x is the input
    image and y is the ground truth mask
    """
    r, c = index
    th, tw = self.tile_size
    h, w = self.mosaic.shape

    # Sample the data using windows
    window = Window(c, r, tw, th)
    x = self.mosaic.read(window=window)

    # If in inference mode and mask doesn't exist, then create dummy label.
    if self.inf_mode and not self.mask_exists:
      y = np.ones((self.num_classes, th, tw))
    else:
      assert self.mask_exists, "Ground truth mask must exist for training."

      # Map values in mask to values within num_classes.
      mask = self.mask.read(window=window)
      mask = np.vectorize(self.map_index)(mask)
      y = CropDataset.one_hot_mask(mask, self.num_classes)

    # Sample is (x, y) pair of image and mask.
    sample = x, y

    # Apply any augmentation.
    if self.transform:
      sample = self.transform(sample)
    
    # Return the sample as tensors.
    to_tensor = lambda t: torch.tensor(t, dtype=torch.float32)

    return to_tensor(sample[0]), to_tensor(sample[1])
  
  def gen_indices(self, indices_path=None):
    """
    Generates indices `(r,c)` corresponding to start position of tiles,
    where the tile is formed by `mosaic[:, r:r+th, c:c+th]`.
    If `indices_path` specified, loads indices from path. If `indices_path`
    is not yet a file on system, then generates and saves the indices.
    """
    if indices_path:
      if os.path.isfile(indices_path):
        with open(indices_path, 'r') as f:
          return json.load(f)

    h, w = self.mosaic.shape
    th, tw = self.tile_size
    step_h = th - self.overlap
    step_w = tw - self.overlap

    # Get (r, c) start position of each tile in area
    inds = []
    for r in range(0, h-step_h, step_h):
      for c in range(0, w-step_w, step_w):
        inds.append((r, c))
    
    # Shuffle em up
    random.shuffle(inds)

    # Add to train/val/test
    indices = {}
    train_split = int(self.train_val_test[0] * len(inds))
    val_split = int(self.train_val_test[1] * len(inds))
    indices["train"] = inds[:train_split]
    indices["val"] = inds[train_split: train_split + val_split]
    indices["test"] = inds[train_split + val_split:]

    # Save if specified.
    if indices_path:
      with open(indices_path, 'w') as f:
        json.dump(indices, f)

    return indices


def get_data_loaders(config_handler, 
                     transform_fn=None,
                     inf_mode=False,
                     train_val_test=[0.8, 0.1, 0.1], 
                     batch_size=32,
                     num_workers=4):
    """
    Creates the train, val and test loaders to input data to the model.
    Specify if you want the loaders for an inference task.
    """
    dataset = CropDataset(config_handler, inf_mode=inf_mode)

    indices = dataset.gen_indices(indices_path=config_handler.indices_path)

    # Define samplers for each of the train, val and test data
    train_sampler = SubsetRandomSampler(indices['train'])
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=train_sampler, num_workers=num_workers)

    val_sampler = SubsetRandomSampler(indices['val'])
    val_loader = DataLoader(dataset, batch_size=batch_size, 
                            sampler=val_sampler, num_workers=num_workers)

    test_sampler = SubsetRandomSampler(indices['test'])
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader
