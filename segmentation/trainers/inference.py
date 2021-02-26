import os
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from typing import Optional

from utils.colors import get_color_choice
from data_loaders.dataset import CropDataset
from metrics import create_metrics_dict, confusion_matrix
from trainers.utils import create_dirs, load_model, create_dataset


class InferenceAgent:
    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            batch_size: int,
            out_dir: str,
            metric_names=(),
    ):
        """
        Handles inference, including saving metrics and TODO: images
        @param model: Segmentation model to evaluate
        @param dataset: CropDataset instance
        @param batch_size: Batch size of input images for inference
        @param out_dir: Output directory where inference results will be saved
        @param metric_names: Names of metrics that will measure inference performance
        """
        self.batch_size = batch_size
        self.out_dir = out_dir

        # Get list of metrics to use for training
        self.metric_names = metric_names
        if len(self.metric_names) == 0:
            self.metric_names = ["iou", "prec", "recall"]

        # Set up dataset
        self.dataset = dataset

        # Set up available device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Get model
        self.model = model
        self.model.to(self.device)

    @classmethod
    def create_inference_agent(
            cls,
            data_path,
            data_map_path,
            out_dir,
            exp_name,
            trainer_config,
            model_config,
            classes,
            checkpoint_path,
            **kwargs
    ):
        """
        Creates a Trainer out of raw arguments
        @param data_path: Path to directory containing all datasets
        @param data_map_path: Path to .json file containing dataset split information
        @param out_dir: Path to directory where training results will be stored
        @param exp_name: Unique name for inference experiment as directory for storing results
        @param trainer_config: JSON config file containing InferenceAgent parameters
        @param model_config: JSON config file containing Model parameters
        @param classes: JSON file containing Cropmap class name --> class id
        @param checkpoint_path: Path to .bin checkpoint file containing model weights
        @param kwargs: Extra keyword arguments to pass to init function.
        @return: Initialised Inference Agent
        """
        # Create output directory, save directory and metrics directories.
        out_dir = os.path.join(
            out_dir if out_dir is not None else os.path.join(data_path, 'inference'),
            exp_name if exp_name is not None else
            "_".join((model_config["name"], trainer_config["name"]))
        )
        create_dirs(out_dir)

        # SAVE config file in output directory at beginning of inference
        cls.save_config(trainer_config, out_dir, 'trainer_config')
        cls.save_config(model_config, out_dir, 'model_config')

        # Set up dataset
        classifier_name = model_config["classifier"].lower()
        interest_classes = trainer_config.get("interest_classes", [])
        transforms = trainer_config.get("transforms", {})
        dataset = cls.create_dataset(classifier_name, data_path, data_map_path,
                                     classes, interest_classes, transforms)

        # TODO: Find a way to break this link between model and trainer config
        # Set up model using its config file and number of classes from trainer config.
        num_classes = len(interest_classes or classes.keys())
        model = load_model(model_config, num_classes, checkpoint_path)

        return cls(
            model=model,
            dataset=dataset,
            batch_size=trainer_config.get("batch_size", 32),
            metric_names=trainer_config.get("metrics", []),
            **kwargs
        )

    @staticmethod
    def create_dataset(classifier_name, data_path, data_map_path,
                       classes, interest_classes, transforms):
        """
        Creates a CropDataset
        @param classifier_name: Name of model classifier, which determines which dataset to use
        @param data_path: Path to directory containing datasets
        @param data_map_path: Path to .json file containing dataset split information
        @param classes: JSON file containing class name --> class id
        @param interest_classes: List of specific classes (subset) to run experiment
        @param transforms: Dictionary of transform names to use for data augmentation
        @return: CropDataset
        """
        return create_dataset(
            classifier_name,
            data_path=data_path,
            data_map_path=data_map_path,
            classes=classes,
            interest_classes=interest_classes,
            transforms=transforms,
            train_val_test=(0.8, 0.1, 0.1),
            use_one_hot=False,
            inf_mode=True
        )

    @staticmethod
    def save_config(config, out_dir, name):
        """
        Saves the config file as a .json file in `out_dir`.
        @param config: JSON config object
        @param out_dir: Output directory where config will be saved as .json file
        @param name: Name of json file
        """
        config_path = os.path.join(out_dir, f"{name}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config file at: {config_path}")

    @staticmethod
    def bytescale(img, high=255):
        """
        Converts an image of arbitrary int dtype to an 8-bit (uint8) image.
        @param img: (h, w, #c) numpy array of any int dtype
        @param high: Max pixel value
        @return: (h, w, #c) numpy array of type uint8
        """
        # Find min and max across each channel
        im = img.reshape(img.shape[-1], -1)
        im_min = np.min(im, axis=1)
        im_max = np.max(im, axis=1)

        scale = 255 / (im_max - im_min)
        im_arr = (img - im_min) * scale

        return im_arr.astype(np.uint8)

    @staticmethod
    def draw_mask_on_im(img, masks):
        """
        Helper method that opens an image, draws the segmentation masks in `masks`
        as bitmaps, and then returns the masked image.
        @param img: np array of shape (h, w, #c)
        @param masks: np array of shape: (#c, h, w)
        @return: image with mask drawn on it as well as raw mask
        """
        # Open the image and set up an ImageDraw object
        im = Image.fromarray(img, mode='RGB')
        im_draw = ImageDraw.Draw(im)

        # Draw the masks in a separate image as well
        raw_mask = Image.fromarray(np.zeros(img.shape[0:2])).convert('RGB')
        mask_draw = ImageDraw.Draw(raw_mask)

        # Draw the bitmap for each class (only if class mask not empty)
        for i, mask in enumerate(masks):
            if mask.any():
                # Create mask, scale its values by 64 so that opacity not too low/high
                mask_arr = mask.astype(np.uint8) * 64
                mask_im = Image.fromarray(mask_arr, mode='L')

                # Get color choice, and draw on image and on raw_mask
                color = get_color_choice(i)
                im_draw.bitmap((0, 0), mask_im, fill=color)
                mask_draw.bitmap((0, 0), mask_im, fill=color)

        return im, raw_mask

    def infer(self, set_type):
        """
        Runs inference for the model on the given set_type for the dataset.
        Saves metrics in inference out directory per ground truth mask.
        @param set_type: One of train/val/test
        """
        train_loaders, val_loaders, test_loaders = \
            self.dataset.create_data_loaders(batch_size=self.batch_size)
        loaders = {'train': train_loaders, 'val': val_loaders, 'test': test_loaders}
        data_loader = loaders[set_type]

        # Begin inference
        for batch_index, (input_t, y) in enumerate(data_loader):
            # Shift to correct device
            input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

            # Input into the model
            preds = self.model(input_t)

            # TODO: Fix inference for time series
            # Convert from tensor to numpy array
            # imgs = input_t.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            label_masks = y.detach().cpu().numpy()

            # TODO: Add back images to this loop later
            # Iterate over each image in batch.
            for ind, (pred, label_mask) in enumerate(zip(preds, label_masks)):
                _pred = pred[np.newaxis, ...]  # shape (b, #c, h, w)
                _label_mask = label_mask[np.newaxis, ...]  # shape (b, #c, h, w)

                # Get raw confusion matrix
                CM = confusion_matrix(_pred, _label_mask, pred_threshold=0)

                # Find the count of each class in ground truth, record in metrics dict as whole num
                n = self.dataset.num_classes
                label_class_counts = np.count_nonzero(label_mask.reshape(n, -1), axis=-1)

                # Create metrics dict
                metrics_dict = create_metrics_dict(
                    self.dataset.remapped_classes,
                    tn=CM[:, 0, 0],
                    fp=CM[:, 0, 1],
                    fn=CM[:, 1, 0],
                    tp=CM[:, 1, 1],
                    gt_class_count=label_class_counts.tolist()
                )

                # Id for saving file.
                img_id = (batch_index * self.batch_size) + ind

                # # Convert from b,g,r (indices/bands 3, 2, 1) to r,g,b
                # # Convert to (h, w, #c) shape and scale to uint8 values
                # img = img[1:4, ...][::-1]
                # img = img.transpose(1, 2, 0)
                # img = self.bytescale(img, high=255)
                #
                # # Save original image
                # im = Image.fromarray(img, mode="RGB")
                # im.save(os.path.join(ch.inf_dir, f"{img_id}_im.jpg"))
                #
                # # Draw pred mask on image
                # pred_im, pred_mask = self.draw_mask_on_im(img, pred > 0)
                # pred_im.save(os.path.join(ch.inf_dir, f"{img_id}_pred.jpg"))
                # pred_mask.save(os.path.join(ch.inf_dir, f"{img_id}_raw_pred.jpg"))
                #
                # # Draw ground truth mask on image
                # gt_im, gt_mask = self.draw_mask_on_im(img, label_mask)
                # gt_im.save(os.path.join(ch.inf_dir, f"{img_id}_gt.jpg"))
                # gt_mask.save(os.path.join(ch.inf_dir, f"{img_id}_raw_gt.jpg"))

                # Save eval results.
                with open(os.path.join(self.out_dir, f"{img_id}_metrics.json"), 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
