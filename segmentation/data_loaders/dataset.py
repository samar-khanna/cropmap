import os
import numpy as np
import torch
import torchvision.transforms as torch_transforms
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
import data_loaders.data_transforms as data_transforms


class CropDataset(Dataset):

    _DATA_MAP_NAME = "data_map"

    def __init__(self, config_handler, data_path, data_map_path=None):
        """
        Abstract class meant to be subclassed for all Cropmap datasets.
        Initialises the class data and the data transforms.

        Requires:
          `config_handler`: Object that handles the model config file.
          `data_path`: Path to dataset directory
          `data_map_path`: Path to .json file containing train/val/test split (optional)
        """
        self.data_path = os.path.abspath(data_path)

        data_map_name = self._DATA_MAP_NAME
        if data_map_path is not None:
            data_map_name = data_map_path.split(os.path.sep)[-1].replace(".json", "")

        self.data_map_path = data_map_path if data_map_path is not None \
            else os.path.join(self.data_path, f"{data_map_name}.json")
        self.indices_path = os.path.join(
            self.data_map_path.replace(f"{data_map_name}.json", ""),
            f"{data_map_name}_indices.json"
        )

        # Filter out interested classes, if specified. Otherwise use all classes.
        # Sort the classes in order of their indices
        classes = config_handler.classes
        interest_classes = config_handler.config.get("interest_classes")
        interest_classes = interest_classes if interest_classes else classes.keys()
        interest_classes = sorted(interest_classes, key=classes.get)

        self.map_class_to_idx, self.map_idx_to_class = \
            CropDataset.create_class_index_map(classes, interest_classes)

        # contains interest_class_name --> label idx in dataset
        self.remapped_classes = {
            class_name: self.map_class_to_idx[classes[class_name]] for class_name in interest_classes
        }
        self.num_classes = len(self.remapped_classes)

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

    def __len__(self):
        raise NotImplementedError("Abstract class. Don't call here.")

    def __getitem__(self, index):
        raise NotImplementedError("Abstract class. Don't call here.")

    def shift_sample_to_device(self, sample, device):
        raise NotImplementedError("Abstract class. Don't call here.")

    def create_data_loaders(self, *args, **kwargs):
        raise NotImplementedError("Abstract class. Don't call here.")

    @staticmethod
    def create_class_index_map(classes, interest_classes):
        """
        Creates 2 maps M (len=max(class_ids)+1) and R (len=len(interest_classes)).
        M[class_id] is the mapped index used to represent class class_id in the dataset.
        R[i] is the class id of the mapped class index i.
        """

        # map_class_to_idx maps from (original class id --> new class index in dataset)
        # map_idx_to_class maps from (new class index --> original class id)
        map_class_to_idx = -1 * np.ones(max(classes.values())+1, dtype=np.int)
        map_idx_to_class = -1 * np.ones(len(interest_classes), dtype=np.int)
        for i, class_name in enumerate(interest_classes):
            map_class_to_idx[int(classes[class_name])] = i
            map_idx_to_class[i] = int(classes[class_name])

        return map_class_to_idx, map_idx_to_class

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

    @staticmethod
    def collate_fn(batch):
        """
        Converts a list of (x,y) samples into batched tensors.
        """
        return default_collate(batch)


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
