import os
import csv
import json
import random
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torchvision.transforms as torch_transforms
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
import data_transforms 


MOSAIC_NAME = "mosaic.tif"
MASK_NAME = "ground_truth.tif"


class ConfigHandler():
  def __init__(self, data_path, path_to_config, classes_path, 
               out_dir=None, inf_subdir=None):
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
    self.out_dir = out_dir if out_dir else \
                   os.path.join(config.get("out_dir", "."), self.name)
    self.save_dir = os.path.join(self.out_dir, "checkpoints")
    self.metrics_dir = os.path.join(self.out_dir, "metrics")
    self.inf_dir = os.path.join(self.out_dir, "inference")
    if inf_subdir:
        self.inf_dir = os.path.join(self.inf_dir, inf_subdir)
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
               train_val_test=[0.8, 0.1, 0.1], inf_mode=False):
    """
    Initialises an instance of a `CropDataset`.
    Requires:
      `config_handler`: Object that handles the model config file.
      `tile_size`: (h,w) denoting size of each tile to sample from area.
      `overlap`: number of pixels adjacent tiles share.
      `train_val_test`: Percentage split (must add to 1) of data sizes
      `inf_mode`: Whether the dataset is being used for inference or not.
                  If not for inference, then make sure that labels exist.
    """
    self.tile_size = tile_size
    self.overlap = overlap
    self.train_val_test = train_val_test
    self.inf_mode = inf_mode

    # Set up label mask pixel mapping based on interested classes.
    self.num_classes = len(config_handler.classes)
    self.map_index = lambda i: config_handler.index_map.get(i, -1)

    # Set up transforms if any
    transforms = getattr(config_handler, "transforms", {})
    composed = []
    for transform_name, transform_kwargs in transforms.items():
      # Get transform function from our own file, or else from torchvision
      transform_class = getattr(data_transforms, transform_name, None)
      if not transform_class:
        transform_class = getattr(torch_transforms, transform_name)
      
      transform_fn = transform_class(**transform_kwargs)
      composed.append(transform_fn)
    
    self.transform = torch_transforms.Compose(composed) if composed else None

    # Dict of files containing each (path_to_mosaic.tif, path_to_mask.tif)
    self.data_paths = {"train":[], "val":[], "test":[]}

    # Check if single or multiple mosaics used for training.
    abs_path = os.path.abspath(config_handler.data_path)
    data_map_path = os.path.join(config_handler.data_path, 'data_map.json')
    if os.path.isfile(data_map_path):

      # Data map specifying which masks to use. Note: this file can repeat
      # names for train/val/test. To separate tiles from the same file, 
      # the train/val/test split will be used.
      with open(data_map_path, 'r') as f:
        self.data_map = json.load(f)
      
      for set_type, data_dirs in self.data_map.items():
        for data_dir in data_dirs:
          rel_path = os.path.relpath(data_dir)
          mosaic_path = os.path.join(abs_path, rel_path, MOSAIC_NAME)
          mask_path = os.path.join(abs_path, rel_path, MASK_NAME)
          if not os.path.isfile(mask_path):
            mask_path = None
          self.data_paths[set_type].append((mosaic_path, mask_path))
    else:
      mosaic_path = os.path.join(abs_path, MOSAIC_NAME)
      mask_path = os.path.join(abs_path, MASK_NAME)
      
      for set_type in ["train", "val", "test"]:
        self.data_paths[set_type].append((mosaic_path, mask_path))

    # Store the shapes for all the mosaic files in the dataset.
    self.mosaic_shapes = {}
    for set_type, paths in self.data_paths.items():
      for (mosaic_path, _) in paths:
        if mosaic_path not in self.mosaic_shapes:
          with rasterio.open(mosaic_path) as mosaic:
            self.mosaic_shapes[mosaic_path] = mosaic.shape

  def __len__(self):
    total_len = 0
    for _, mosaic_shape in self.mosaic_shapes.items():
      h, w = mosaic_shape
      th, tw = self.tile_size
      o = self.overlap
      n_rows = h//(th - o)
      n_cols = w//(tw - o)
      total_len += n_rows * n_cols
    return total_len
  
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
    y = y * np.arange(num_classes)
    y = y.T

    # One hot encoding by checking for equality with dense mask
    y = y == mask

    return y

  def __getitem__(self, index):
    """
    This returns only ONE sample from the dataset, for a given index.
    The index is a tuple of the form `(set_type, data_paths_index, (row, col))`
    The returned sample should be a tuple `(x, y)` where `x` is the input image 
    of shape `(#bands, h, w)` and `y` is the ground truth mask of shape `(c, h, w)`
    """
    set_type, i, (r, c) = index
    th, tw = self.tile_size

    # Access the right mosaic and mask file paths
    mosaic_path, mask_path = self.data_paths[set_type][i]

    # Sample the data using windows
    window = Window(c, r, tw, th)
    with rasterio.open(mosaic_path) as mosaic:
      x = mosaic.read(window=window)

    # If in inference mode and mask doesn't exist, then create dummy label.
    if self.inf_mode and not mask_path:
      y = np.ones((self.num_classes, th, tw))
    else:
      assert mask_path, "Ground truth mask must exist for training."

      with rasterio.open(mask_path) as _mask:
        mask = _mask.read(window=window)

      # Map values in mask to values within num_classes.
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

  @staticmethod
  def convert_inds(_indices):
    """
    Converts a given dictionary of data indices into the correct format
    required by the Sampler. Concretely, converts a dictionary of the form
    `{set_type: {data_paths_index: [(r1, c1), ...]}}` to a list of the form
    `[(set_type, data_paths_index, (r1, c1)), ...]`
    """
    indices = {"train":[], "val":[], "test":[]}
    for set_type, data_paths_dict in _indices.items():
      for data_path_ind, offsets in data_paths_dict.items():
        for (r, c) in offsets:
          new_ind = (set_type, int(data_path_ind), (r, c))
          indices[set_type].append(new_ind)
    return indices
  

  def gen_indices(self, indices_path=None):
    """
    Generates indices `(r,c)` corresponding to start position of tiles,
    where the tile is formed by `mosaic[:, r:r+th, c:c+tw]`.
    If `indices_path` specified, loads indices from path. If `indices_path`
    is not yet a file on system, then generates and saves the indices.
    """
    if indices_path:
      if os.path.isfile(indices_path):
        with open(indices_path, 'r') as f:
          _indices = json.load(f)
        return CropDataset.convert_inds(_indices)

    # Note which sets each mosaic appears in. Mapping from
    # mosaic_path -> {"train": (data_path_ind, split_pct), "val":...}
    # If appears in 2/3 sets, extra given in order of preferernce [train, val, test]
    mosaic_splits = {}
    for i, (mosaic_path, _) in enumerate(self.data_paths["train"]):
      mosaic_splits[mosaic_path] = {"train": {"ind": i, "pct":1.0}}
    
    for i, (mosaic_path, _) in enumerate(self.data_paths["val"]):
      splits = mosaic_splits.get(mosaic_path, {})
      splits["val"] = {"ind": i}
      if "train" in splits:
        # Remove percentage of val indices from training data
        splits["train"]["pct"] -= self.train_val_test[1]  # 1-val = tr + te
        splits["val"]["pct"] = self.train_val_test[1]
      else:
        splits["val"]["pct"] = 1.0
      mosaic_splits[mosaic_path] = splits
    
    for i, (mosaic_path, _) in enumerate(self.data_paths["test"]):
      splits = mosaic_splits.get(mosaic_path, {})
      splits["test"] = {"ind": i}
      # Either train present, or train, val present, or just val
      if "train" in splits:
        splits["train"]["pct"] -= self.train_val_test[2]  # 1-te-val = tr
        splits["test"]["pct"] = self.train_val_test[2]
      # If just val and test are used, then split between val and test equally
      elif "val" in splits:
        splits["val"]["pct"] = splits["test"]["pct"] = 0.5
      else:
        splits["test"]["pct"] = 1.0
      mosaic_splits[mosaic_path] = splits

    # Create the (row, col) index offsets.
    indices = {}
    for mosaic_path, mosaic_shape in self.mosaic_shapes.items():
      h, w = mosaic_shape
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

      # Split them into each train/val/test, as specified
      splits = mosaic_splits[mosaic_path]
      prev_split = 0
      for set_type, ind_split in splits.items():
        ind, pct = ind_split["ind"], ind_split["pct"]
        split = int(pct * len(inds)) + prev_split

        set_indices = indices.get(set_type, {})
        set_indices[ind] = inds[prev_split: split]
        indices[set_type] = set_indices

        prev_split = split

    # Save if specified.
    if indices_path:
      with open(indices_path, 'w') as f:
        json.dump(indices, f, indent=2)

    return CropDataset.convert_inds(indices)


class SubsetSequentialSampler(Sampler):
  """
  Almost identical to `SubsetRandomSampler` except it samples sequentially 
  from given indices (i.e. always in the same order).
  """
  def __init__(self, indices):
    self.indices = indices
  
  def __iter__(self):
    for ind in self.indices:
      yield ind
    
  def __len__(self):
    return len(self.indices)


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
  # Sample train data randomly, and validation, test data sequentially
  train_sampler = SubsetRandomSampler(indices['train'])
  train_loader = DataLoader(dataset, batch_size=batch_size, 
                            sampler=train_sampler, num_workers=num_workers)

  val_sampler = SubsetSequentialSampler(indices['val'])
  val_loader = DataLoader(dataset, batch_size=batch_size, 
                          sampler=val_sampler, num_workers=num_workers)

  test_sampler = SubsetSequentialSampler(indices['test'])
  test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

  return train_loader, val_loader, test_loader
