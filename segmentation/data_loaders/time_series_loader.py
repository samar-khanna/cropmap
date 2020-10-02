import os
import csv
import json
import random
import rasterio
from rasterio.windows import Window
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from data_loaders.dataset import CropDataset, SubsetSequentialSampler


MOSAIC_NAME = "mosaic.tif"
MASK_NAME = "ground_truth.tif"
INDICES_FILE_NAME = "time_series_indices.json"
DATA_MAP_NAME = "time_series_map.json"


TimeSeriesSample = namedtuple("TimeSeriesSample", ["inputs", "label"])


class TimeSeriesDataset(CropDataset):
    def __init__(self, config_handler, tile_size=(224, 224), overlap=0,
                 inf_mode=False, **kwargs):
        """
        Initialises an instance of a TimeSeriesDataset for sequences of images.
        Expects format to be multi-input, single-output.
        ASSUMES: Label for all inputs in time sequence is same, and is found in any
                 of the input's directory.

        Requires:
          `config_handler`: Object that handles the model config file.
          `tile_size`: (h,w) denoting size of each tile to sample from area.
          `overlap`: number of pixels adjacent tiles share.
          `inf_mode`: Whether the dataset is being used for inference or not.
                      If not for inference, then make sure that labels exist.
        """
        super().__init__(config_handler)

        self.tile_size = tile_size
        self.overlap = overlap
        self.inf_mode = inf_mode

        abs_path = os.path.abspath(config_handler.data_path)

        # Indices according to train/val/test will be stored here
        self.indices_path = os.path.join(abs_path, INDICES_FILE_NAME)

        # Dict of files containing TimeSeriesSample objects
        # ([path_to_mosaic1.tif, path_to_mosaic2.tif, ...], path_to_label_mask.tif)
        self.data_paths = {"train": [], "val": [], "test": []}

        # Format of this file must be: set_type --> [sequence1, sequence2, ...]
        # Each sequence_i := [path/to/data_dir1, path/to/data_dir2]
        data_map_path = os.path.join(config_handler.data_path, DATA_MAP_NAME)
        assert os.path.isfile(data_map_path), "Require data map for time-series data"

        # Data map specifying how to group inputs in sequences and associate labels
        with open(data_map_path, 'r') as f:
            self.data_map = json.load(f)

        for set_type, sequence_list in self.data_map.items():
            # Assumes same label for all data in a time-sequence
            for time_sequence in sequence_list:
                rel_path = os.path.relpath(time_sequence[0])
                mask_path = os.path.join(abs_path, rel_path, MASK_NAME)
                if not os.path.isfile(mask_path):
                    mask_path = None

                sample = TimeSeriesSample(
                    inputs=[os.path.join(abs_path, os.path.relpath(data_dir), MOSAIC_NAME)
                            for data_dir in time_sequence],
                    label=mask_path
                )
                self.data_paths[set_type].append(sample)

        # Store the shapes for all the mosaic files in the dataset.
        self.mosaic_shapes = {}
        for set_type, samples in self.data_paths.items():
            for sample in samples:
                for mosaic_path in sample.inputs:
                    if mosaic_path not in self.mosaic_shapes:
                        with rasterio.open(mosaic_path, 'r') as mosaic:
                            self.mosaic_shapes[mosaic_path] = mosaic.shape

    def __len__(self):
        # WARNING: This is full possible length of dataset, not cleaned.
        total_len = 0
        for _, mosaic_shape in self.mosaic_shapes.items():
            h, w = mosaic_shape
            th, tw = self.tile_size
            o = self.overlap
            n_rows = h//(th - o)
            n_cols = w//(tw - o)
            total_len += n_rows * n_cols
        return total_len

    def __getitem__(self, index):
        """
        This returns only ONE sample from the dataset, for a given index.
        The index is a tuple of the form `(set_type, data_paths_index, (row, col))`
        The returned sample should be a tuple `(x, y)` where `x` is the input image
        of shape `(#bands, h, w)` and `y` is the ground truth mask of shape `(c, h, w)`
        """
        set_type, i, (r, c) = index
        th, tw = self.tile_size
        window = Window(c, r, tw, th)

        # Access the right sample file paths
        time_sample = self.data_paths[set_type][i]
        mask_path = time_sample.label

        x_series = []
        for mosaic_path in time_sample.inputs:
            # Sample the data using same window (same region)
            with rasterio.open(mosaic_path, 'r') as mosaic:
                x = mosaic.read(window=window)
            x_series.append(x)

        # If in inference mode and mask doesn't exist, then create dummy label.
        if self.inf_mode and not mask_path:
            y = np.ones((self.num_classes, th, tw))
        else:
            assert mask_path is not None, "Ground truth mask must exist for training."

            with rasterio.open(mask_path, 'r') as _mask:
                mask = _mask.read(window=window)

            # Map class ids in mask to indexes within num_classes.
            mask = self.map_class_to_idx[mask]
            y = TimeSeriesDataset.one_hot_mask(mask, self.num_classes)

        # Sample is ([x1, ..., xt], y) pair of image sequence and mask.
        sample = x_series, y

        # Apply any augmentation.
        if self.transform:
            sample = self.transform(sample)

        # Return the sample as tensors.
        to_tensor = lambda t: torch.as_tensor(t, dtype=torch.float32)

        return [to_tensor(x) for x in sample[0]], to_tensor(sample[1])

    @staticmethod
    def convert_inds(_indices):
        """
        Converts a given dictionary of data indices into the correct format
        required by the Sampler. Concretely, converts a dictionary of the form
        `{set_type: {data_paths_index: [(r1, c1), ...]}}` to a list of the form
        `[(set_type, data_paths_index, (r1, c1)), ...]`
        """
        indices = {"train": [], "val": [], "test": []}
        for set_type, data_paths_dict in _indices.items():
            for data_path_ind, offsets in data_paths_dict.items():
                for (r, c) in offsets:
                    new_ind = (set_type, int(data_path_ind), (r, c))
                    indices[set_type].append(new_ind)
        return indices

    def gen_indices(self, regen_indices=False, clean_indices=True):
        """
        Generates indices `(r,c)` corresponding to start position of tiles,
        where the tile is formed by `mosaic[:, r:r+th, c:c+tw]`.
        If `indices_path` specified, loads indices from path. If `indices_path`
        is not yet a file on system, then generates and saves the indices.
        """
        if os.path.isfile(self.indices_path) and not regen_indices:
            with open(self.indices_path, 'r') as f:
                _indices = json.load(f)
            return TimeSeriesDataset.convert_inds(_indices)

        print("Generating indices...")

        # Before convert_inds, is of form: set_type --> {time_sample_index: [inds]}
        indices = {}
        for set_type, time_samples in self.data_paths.items():
            for sample_index, time_sample in enumerate(time_samples):
                # Require same shape for all mosaics in a time series
                sample_shape = self.mosaic_shapes[time_sample.inputs[0]]
                for mosaic_path in time_sample.inputs:
                    mosaic_shape = self.mosaic_shapes[mosaic_path]
                    assert mosaic_shape == sample_shape, \
                        f"Shape {mosaic_shape} of {mosaic_path} doesn't match with {sample_shape}"

                h, w = sample_shape
                th, tw = self.tile_size
                step_h = th - self.overlap
                step_w = tw - self.overlap

                # Get (r, c) start position of each tile in area
                inds = [(r, c) for r in range(0, h-step_h, step_h) for c in range(0, w-step_w, step_w)]

                # Clean by rejecting indices of any x in time sample that contain NaN
                if clean_indices:
                    def is_not_nan(position):
                        x_list, y = self.__getitem__([(set_type, sample_index, position)])
                        for x in x_list:
                            if torch.isnan(x).any():
                                return False
                        return True
                    inds = list(filter(is_not_nan, inds))

                # Shuffle em up
                random.shuffle(inds)

                # Add to dict of indices
                set_indices = indices.get(set_type, {})
                set_indices[sample_index] = inds
                indices[set_type] = set_indices

        # Save indices
        with open(self.indices_path, 'w') as f:
            json.dump(indices, f, indent=2)

        print(f"Done creating indices at {self.indices_path}")
        return TimeSeriesDataset.convert_inds(indices)

    def create_data_loaders(self, regen_indices=False, batch_size=32, num_workers=4):
        """
        Creates the train, val and test loaders to input data to the model.
        Specify if you want the loaders for an inference task.
        """
        # Generates indices if not present
        indices = self.gen_indices(regen_indices=regen_indices)

        # Define samplers for each of the train, val and test data
        # Sample train data randomly, and validation, test data sequentially
        train_sampler = SubsetRandomSampler(indices['train'])
        train_loader = DataLoader(self, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=num_workers)

        val_sampler = SubsetSequentialSampler(indices['val'])
        val_loader = DataLoader(self, batch_size=batch_size,
                                sampler=val_sampler, num_workers=num_workers)

        test_sampler = SubsetSequentialSampler(indices['test'])
        test_loader = DataLoader(self, batch_size=batch_size,
                                 sampler=test_sampler)

        return train_loader, val_loader, test_loader
