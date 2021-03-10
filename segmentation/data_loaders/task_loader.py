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
from data_loaders.time_series_loader import TimeSeriesDataset
from data_loaders.dataset import CropDataset, SubsetSequentialSampler


MOSAIC_NAME = "mosaic.tif"
MASK_NAME = "ground_truth.tif"
DATA_MAP_NAME = "task_image_map"


Splits = namedtuple("Splits", ["train", "val", "test"], defaults=[0.8, 0.1, 0.1])

TaskInfo = namedtuple(
    "TaskInfo",
    ["task_name", "set_type", "mosaic_paths", "mask_path", "support_pct", "query_pct"],
    defaults=[0.5, 0.5]
)


class TaskDataset(CropDataset):

    _DATA_MAP_NAME = DATA_MAP_NAME

    def __init__(self,
                 data_path, classes, interest_classes=(), data_map_path=None, transforms=None,
                 tile_size=(224, 224), overlap=0, train_val_test=Splits(), use_one_hot=True,
                 inf_mode=False, **kwargs):
        super().__init__(data_path, classes, interest_classes, data_map_path, transforms)

        self.tile_size = tile_size
        self.overlap = overlap
        self.use_one_hot = use_one_hot
        self.inf_mode = inf_mode
        self.train_val_test = train_val_test

        # Format of this file must be: set_type --> [task1_dict, task2_dict, ...]
        # where taski_dict := {task_name: __, task_paths: __, support_pct: __}
        # Note: The same mosaic can be repeated for train/val/test (in which case the splits
        # from train/val/test will be used)
        # But mosaics cannot be repeated within a set_type across tasks.
        assert os.path.isfile(self.data_map_path), "Require data map for task data"
        with open(self.data_map_path, 'r') as f:
            self.data_map = json.load(f)

        # Dict of tasks {task1_name: task1_info, task2_name: task2_info, ...}
        self.data_split = {}

        for set_type, tasks in self.data_map.items():
            all_task_paths = set()
            for i, task in enumerate(tasks):
                assert "task_paths" in task, \
                    "Need to specify path(s) to directory(ies) containing mask + mosaic."

                # Task name is unique identifier (eg: train_cali_aug_07)
                task_name = f"{set_type}_t{i}_" + task.get("task_name", "")
                support_pct = task.get("support_pct", 0.5)
                query_pct = 1.0 - support_pct

                rel_mask_path = os.path.relpath(task['task_paths'][0])
                mask_path = os.path.join(self.data_path, rel_mask_path, MASK_NAME)
                if not os.path.isfile(mask_path) and self.inf_mode:
                    mask_path = None

                mosaic_paths = []
                for task_path_dir in task['task_paths']:
                    rel_path = os.path.relpath(task_path_dir)
                    assert rel_path not in all_task_paths, \
                        f"Cannot reuse same mosaic for multiple tasks within {set_type} set"
                    all_task_paths.add(rel_path)

                    mosaic_paths.append(os.path.join(self.data_path, rel_path, MOSAIC_NAME))

                task_info = TaskInfo(
                    task_name=task_name, set_type=set_type,
                    mosaic_paths=mosaic_paths, mask_path=mask_path,
                    support_pct=support_pct, query_pct=query_pct
                )
                self.data_split[task_name] = task_info

        # Store the shapes for all the mosaic files in the dataset.
        self.mosaic_shapes = {}
        for task_name, task_info in self.data_split.items():
            mosaic_paths = task_info.mosaic_paths
            if tuple(mosaic_paths) not in self.mosaic_shapes:
                seq_h, seq_w = float('inf'), float('inf')
                for mosaic_path in mosaic_paths:
                    with rasterio.open(mosaic_path, 'r') as mosaic:
                        h, w = mosaic.shape
                        seq_h, seq_w = min(seq_h, h), min(seq_w, w)

                self.mosaic_shapes[tuple(mosaic_paths)] = (seq_h, seq_w)

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

    def __getitem__(self, index):
        task_name, (r, c) = index
        th, tw = self.tile_size

        # Access the right mosaic and mask file paths
        task_info = self.data_split[task_name]
        mask_path = task_info.mask_path

        x_series = []
        for mosaic_path in task_info.mosaic_paths:
            # Sample the data using same window (same region)
            x = self.read_window(mosaic_path, c, r, tw, th)
            x_series.append(x)

        # If in inference mode and mask doesn't exist, then create dummy label.
        if self.inf_mode and not mask_path:
            y = np.ones((self.num_classes, th, tw))
        else:
            assert mask_path, "Ground truth mask must exist for training."

            mask = self.read_window(mask_path, c, r, tw, th)

            # Map class ids in mask to indexes within num_classes.
            mask = self.map_class_to_idx[mask]
            y = self.one_hot_mask(mask, self.num_classes) if self.use_one_hot else mask

        # Sample is (x, y) pair of image and mask.
        sample = x_series, y

        # Apply any augmentation.
        if self.transform:
            sample = self.transform(sample)

        # Return the sample as tensors.
        to_tensor = lambda t: torch.tensor(t, dtype=torch.float32)

        return [to_tensor(x) for x in sample[0]], to_tensor(sample[1])

    def shift_sample_to_device(self, sample, device):
        """
        Shifts tensors in sample to device.
        """
        x_list, y = sample
        return [x.to(device) for x in x_list], y.to(device)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate for time-series to handle variable length sequences.
        Pads shorter time series inputs with -1 so that no data is lost.
        @param batch: [(series1, y1), (series2, y2), ...]
        @return:
        """
        return TimeSeriesDataset.collate_fn(batch)

    @staticmethod
    def convert_inds(_indices):
        indices = {"train": {}, "val": {}, "test": {}}
        for set_type, tasks_for_set in _indices.items():
            for task_name, task_indices in tasks_for_set.items():
                indices[set_type][task_name] = {}
                # task_phase is one of support/query
                for task_phase, offsets in task_indices.items():
                    indices[set_type][task_name][task_phase] = []
                    for (r, c) in offsets:
                        new_ind = (task_name, (r, c))
                        indices[set_type][task_name][task_phase].append(new_ind)
        return indices

    def gen_indices(self, regen_indices=False, clean_indices=True):
        if os.path.isfile(self.indices_path) and not regen_indices:
            with open(self.indices_path, 'r') as f:
                _indices = json.load(f)
            return self.convert_inds(_indices)

        print("Generating indices...")

        # Contains info about the set_type (train/val/test) split percentage and task
        SplitInfo = namedtuple("SplitInfo", ["set_split", "task_info"])
        set_priority = {"train": 0, "val": 1, "test": 2}

        # First record mosaic_path -> List of tasks it appears in
        mosaic_splits = {}
        for task_name, task_info in self.data_split.items():
            mosaic_paths = tuple(task_info.mosaic_paths)
            tasks = mosaic_splits.get(mosaic_paths, [])
            tasks.append(task_info)
            mosaic_splits[mosaic_paths] = tasks

        # Now calculate percentage of mosaic that goes to train/val/test
        # mosaic_splits now records mosaic_path -> dictionary of split,
        # task information per train/val/test
        for mosaic_paths, tasks in mosaic_splits.items():
            assert 1 <= len(tasks) <= 3, "At least 1 & at most 3 tasks per mosaic."
            # Order tasks by train/val/test (only 1 task per set_type)
            sorted_tasks = sorted(tasks, key=lambda t: set_priority.get(t.set_type))
            first_task = sorted_tasks[0]

            # Dict mapping {set_type: SplitInfo}
            set_splits = {first_task.set_type: SplitInfo(set_split=1.0, task_info=first_task)}

            # Take away percentage from highest priority set_type present
            for task_info in sorted_tasks[1:]:
                set_type = task_info.set_type
                set_split_pct = self.train_val_test._asdict()[set_type]
                set_splits[set_type] = SplitInfo(set_split=set_split_pct, task_info=task_info)

                # Replace percentage of first split: eg: 1-val = tr+te
                first_split = set_splits[first_task.set_type]
                new_split = first_split._replace(set_split=first_split.set_split - set_split_pct)
                set_splits[first_task.set_type] = new_split

            mosaic_splits[mosaic_paths] = set_splits

        # Create the (row, col) index offsets.
        indices = {}
        for mosaic_path_seq, mosaic_shape in self.mosaic_shapes.items():
            h, w = mosaic_shape
            th, tw = self.tile_size

            # Get (r, c) start position of each tile in area, and shuffle
            inds = [(r, c) for r in range(0, h - th, th) for c in range(0, w - tw, tw)]
            random.shuffle(inds)

            # Dict mapping {set_type: SplitInfo}
            # Split into each train/val/test, and then support/query, as specified
            set_splits = mosaic_splits[mosaic_path_seq]
            prev_split = 0
            for set_type, split_info in set_splits.items():
                task_name = split_info.task_info.task_name
                pct = split_info.set_split
                support_pct = split_info.task_info.support_pct

                # Clean by rejecting indices of any x in time sample that contain NaN
                if clean_indices:
                    def is_not_nan(position):
                        x_list, y = self.__getitem__((task_name, position))
                        for x in x_list:
                            if torch.isnan(x).any():
                                return False
                        return True

                    inds = list(filter(is_not_nan, inds))

                curr_split = int(pct * len(inds)) + prev_split
                set_inds = inds[prev_split: curr_split]
                support_split = int(support_pct * len(set_inds)) if set_type == "train" else 0

                tasks_for_set = indices.get(set_type, {})
                tasks_for_set[task_name] = {
                    "support": set_inds[:support_split],
                    "query": set_inds[support_split:]
                }

                # Update the indices
                indices[set_type] = tasks_for_set
                prev_split = curr_split

        # Save indices
        with open(self.indices_path, 'w') as f:
            json.dump(indices, f, indent=2)

        print(f"Done creating indices at {self.indices_path}")
        return self.convert_inds(indices)

    def create_data_loaders(self, regen_indices=False, batch_size=32, num_workers=4):
        # Generates indices if not present
        indices = self.gen_indices(regen_indices=regen_indices)

        # Sample train data randomly, and validation, test data sequentially
        train_loaders = {
            task_name: [
                DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                           sampler=SubsetRandomSampler(inds), collate_fn=self.collate_fn)
                for task_type, inds in task_inds.items()  # task_type is support/query
            ]
            for task_name, task_inds in indices['train'].items()
        }

        val_loaders = {
            task_name: DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                                  sampler=SubsetSequentialSampler(task_inds['query']),
                                  collate_fn=self.collate_fn)
            for task_name, task_inds in indices['val'].items()
        }

        test_loaders = {
            task_name: DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                                  sampler=SubsetSequentialSampler(task_inds['query']),
                                  collate_fn=self.collate_fn)
            for task_name, task_inds in indices['test'].items()
        }

        return train_loaders, val_loaders, test_loaders






