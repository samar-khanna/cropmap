import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import torchvision.transforms as torch_transforms

from utils.colors import get_cmap
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
import data_loaders.data_transforms as data_transforms


class CropDataset(Dataset):

    _DATA_MAP_NAME = "data_map"

    def __init__(
            self,
            data_path,
            classes,
            interest_classes=(),
            data_map_path=None,
            use_one_hot=True,
            transforms=None
    ):
        """
        Abstract class meant to be subclassed for all datasets.
        Initialises the class data and the data transforms.

        @param data_path: Path to directory containing dataset
        @param classes: Dict of {class_name: class_id}
        @param interest_classes: List of class names to use (subset of all classes)
        @param data_map_path: Path to .json file containing train/val/test splits
        @param use_one_hot: Whether the mask will use one-hot encoding or class id per pixel
        @param transforms: Sequence of transform names available in `data_transforms` file
        """
        self.use_one_hot = use_one_hot
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

        # Load all classes.
        self.all_classes = classes

        # Filter out interested classes, if specified. Otherwise use all classes.
        # Sort the classes in order of their indices
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
        transforms = transforms if transforms is not None else {}
        composed = []
        for transform_name, transform_kwargs in transforms.items():
            # Get transform function from our own file, or else from torchvision
            transform_class = getattr(data_transforms, transform_name, None)
            if transform_class is None:
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
        @param mask: (1, h, w) dense mask
        @param num_classes: Total number of classes (i.e. c)
        @return: (c, h, w) one-hot mask (the one-hot is along axis 0)
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
    def inverse_one_hot_mask(one_hot_mask, unk_value=-1):
        """
        Inverse operation of one_hot_mask. Returns the dense mask given
        a one-hot encoded map
        @param one_hot_mask: (c, h, w) one-hot mask. Each 2d (h,w) slice corresponds
        to class mask associated with that slice's index (i.e. from 0 to c-1)
        @param unk_value: Value to put int dense mask for pixels belonging to no class
        @return: (1, h, w) dense mask, each known pixel is value between 0 to c-1
        """
        mask = np.argmax(one_hot_mask, axis=0)  # (c, h, w) -> (h, w)
        unk_mask = np.all(~one_hot_mask.astype(np.bool), axis=0)  # (c, h, w) -> (h, w)
        mask[unk_mask] = unk_value
        return np.expand_dims(mask, axis=0)  # (1, h, w)

    def format_for_display(self, pred, gt):
        """
        Colorises a model prediction and ground truth mask according to crop colors.
        @param pred: (c, h, w) np array of predicted class logits/probabilities per pixel
        @param gt: (c, h, w) or (1, h, w) np array of ground truth one-hot or class index
        @return: (2, 3, h, w) np array of RGB colored pred & gt
        """
        if self.use_one_hot:
            gt = self.inverse_one_hot_mask(gt)  # (c, h, w) -> (1, h, w)
        gt = np.squeeze(gt, axis=0)  # (1, h, w) -> (h, w)
        unk_mask = gt == -1  # (h, w)

        pred = np.argmax(pred, axis=0)  # (c, h, w) -> (h, w)

        # Map to original class idx
        pred = self.map_idx_to_class[pred]  # (h, w)
        pred[unk_mask] = -1
        gt = self.map_idx_to_class[gt]  # (h, w)
        gt[unk_mask] = -1

        # Colorise the images and drop alpha channel
        cmap = get_cmap(self.all_classes)
        color_gt = cmap(gt).transpose(2, 0, 1)  # (h,w) -> (h,w,4) -> (4,h,w)
        color_pred = cmap(pred).transpose(2, 0, 1)  # (h,w) -> (h,w,4) -> (4,h,w)

        # Drop alpha channel
        display_im = np.stack((color_pred[:3, ...], color_gt[:3, ...]), axis=0)
        return display_im  # (2, 3, h, w)

    @staticmethod
    def read_window(path_to_tif, col: int, row: int, width: int, height: int):
        """
        Reads a section of a .tif file given by the col, row, width and height.
        @param path_to_tif: Path to .tif file to be read
        @param col: Column offset
        @param row: Row offset
        @param width: Width of window to read
        @param height: Height of window to read
        @return: Numpy array of resulting window/tile of .tif file.
        """
        window = Window(col, row, width, height)
        with rasterio.open(path_to_tif, 'r') as fd:
            arr = fd.read(window=window)
        return arr

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
