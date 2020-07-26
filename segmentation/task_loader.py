import os
import csv
from collections import defaultdict
import json
import random
from functools import partial

import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torchvision.transforms as torch_transforms
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler

import data_transforms 


MOSAIC_NAME = "mosaic.tif"
MASK_NAME = "ground_truth.tif"



class TaskDistribution:
  def __init__(self, config, inf_mode=False, support_frac=.5, tile_size=(256,256)):
    abs_path = os.path.abspath(config.data_path)
    data_map_path = os.path.join(config.data_path, 'data_map.json')

    with open(data_map_path, 'r') as f:
      self.data_map = json.load(f)

    self.tasks = defaultdict(list)
    
    for metaset_type, data_dirs in self.data_map.items():
      for data_dir in data_dirs:
        rel_path = os.path.relpath(data_dir)
        mosaic_path = os.path.join(abs_path, rel_path, MOSAIC_NAME)
        mask_path = os.path.join(abs_path, rel_path, MASK_NAME)
        if not os.path.isfile(mask_path):
          mask_path = None
        self.tasks[metaset_type].append(
            Task(mosaic_path, mask_path, config, inf_mode, support_frac,
                  tile_size)
        )

    for metaset_type, _ in self.tasks.items():
      random.shuffle(self.tasks[metaset_type])


  def __getitem__(self, key):
    assert key in ['metatrain', 'metaval', 'metatest'], f'{key} invalid'
    return self.tasks[key]



class Task:
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


  def __init__(self, mosaic_path, mask_path, config, inf_mode, support_frac, tile_size):
    """
    Initialises an instance of a `Task`.
    Requires:
      `config`: Object that handles the model config file.
      `tile_size`: (h,w) denoting size of each tile to sample from area.
      `inf_mode`: Whether the dataset is being used for inference or not.
                  If not for inference, then make sure that labels exist.
    """
    self.tile_size = tile_size
    self.inf_mode = inf_mode
    self.support_frac = support_frac
    self.mosaic_path = mosaic_path
    self.mask_path = mask_path

    # Set up label mask pixel mapping based on interested classes.
    self.num_classes = len(config.classes)
    self.map_index = lambda i: config.index_map.get(i, -1)

    # Set up transforms if any
    transforms = getattr(config, "transforms", {})
    composed = []
    for transform_name, transform_kwargs in transforms.items():
      # Get transform function from our own file, or else from torchvision
      transform_class = getattr(data_transforms, transform_name, None)
      if not transform_class:
        transform_class = getattr(torch_transforms, transform_name)
      
      transform_fn = transform_class(**transform_kwargs)
      composed.append(transform_fn)
    
    self.transform = torch_transforms.Compose(composed) if composed else None

    with rasterio.open(mosaic_path, 'r') as mosaic:
      self.mosaic_shape = mosaic.shape

    # generate/read support and query indexes
    self.support_inds, self.query_inds = \
                    self._gen_indices(indices_path=config.indices_path)


  def _gen_indices(self, indices_path=None):
    """
    Generates indices `(r,c)` corresponding to start position of tiles,
    where the tile is formed by `mosaic[:, r:r+th, c:c+tw]`.
    If `indices_path` specified, loads indices from path. If `indices_path`
    is not yet a file on system, then generates and saves the indices.
    """
    """
    if indices_path and os.path.isfile(indices_path):
      with open(indices_path, 'r') as f:
        _on_disk = json.load(f)
        if self.mosaic_path in _on_disk.keys():
          return (_on_disk[self.mosaic_path]['query'],
                  _on_disk[self.mosaic_path]['support'])
    """

    # Create the (row, col) index offsets.
    indices = []
    h, w = self.mosaic_shape
    th, tw = self.tile_size

    # Get (r, c) start position of each tile in area
    inds = []
    for r in range(0, h-th, th):
      for c in range(0, w-tw, tw):
        inds.append((r, c))
    
    # Shuffle em up
    random.shuffle(inds)
    split_ind = int(self.support_frac*len(inds))
    support_inds = inds[:split_ind]
    query_inds = inds[split_ind:]

    # Save if specified.
    """
    if indices_path:
      mode = 'r+' if os.path.exists(indices_path) else 'w'
      with open(indices_path, mode) as f:
        if mode == 'r+':
          on_disk_inds = json.load(f)
          on_disk_inds[self.mosaic_path] = {
            'query': query_inds, 
            'support': support_inds
          }
        else:
          on_disk_inds = {
            self.mosaic_path: {'query': query_inds, 'support': support_inds}
          }

        json.dump(on_disk_inds, f, indent=2)
    """

    return support_inds, query_inds


  def __getitem__(self, index):
    """
    This returns only ONE sample from the dataset, for a given index.
    The index is a tuple of the form `(set_type, data_paths_index, (row, col))`
    The returned sample should be a tuple `(x, y)` where `x` is the input image 
    of shape `(#bands, h, w)` and `y` is the ground truth mask of shape `(c, h, w)`
    """
    (r, c) = index
    th, tw = self.tile_size

    # Sample the data using windows
    window = Window(c, r, tw, th)
    with rasterio.open(self.mosaic_path, 'r') as mosaic:
      x = mosaic.read(window=window)

    # If in inference mode and mask doesn't exist, then create dummy label.
    if self.inf_mode and not mask_path:
      y = np.ones((self.num_classes, th, tw))
    else:
      assert self.mask_path, "Ground truth mask must exist for training."

      with rasterio.open(self.mask_path, 'r') as _mask:
        mask = _mask.read(window=window)

      # Map values in mask to values within num_classes.
      mask = np.vectorize(self.map_index)(mask)
      y = Task.one_hot_mask(mask, self.num_classes)

    # Sample is (x, y) pair of image and mask.
    sample = x, y

    # Apply any augmentation.
    if self.transform:
      sample = self.transform(sample)
    
    # Return the sample as tensors.
    to_tensor = lambda t: torch.tensor(t, dtype=torch.float32)

    return to_tensor(sample[0]), to_tensor(sample[1])
 

def _gen_task_loaders(batch_size, num_workers, tasks):
  support_sampler = SubsetRandomSampler(task.support_inds)
  query_sampler = SubsetRandomSampler(task.query_inds)

  query_loader = DataLoader(task, batch_size=batch_size, 
                            sampler=query_sampler, num_workers=num_workers)
  support_loader = DataLoader(task, batch_size=batch_size,
                            sampler=support_sampler, num_workers=num_workers)
  return query_loader, support_loader


def get_task_loaders(config, inf_mode=False, batch_size=32, num_workers=4):
  tasks = TaskDistribution(config, inf_mode)
  task_loaders = {}
  func = partial(_gen_task_loaders, batch_size, num_workers)
  for metaset_type in ['metatrain', 'metatest', 'metaval']:
    task_loaders[metaset_type] = map(func, tasks[metaset_type])
  return task_loaders
