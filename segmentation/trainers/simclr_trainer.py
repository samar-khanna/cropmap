import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from random import random, randint
from copy import copy

from trainers.trainer import Trainer
from data_loaders.dataset import CropDataset
from data_loaders import data_transforms
from metrics import calculate_metrics, MeanMetric
from models.loss import BatchCriterion
from trainers.utils import save_model

class SimCLRTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            loss_fn: nn.Module,
            optim_class: optim.Optimizer,
            num_shots: Optional[int],
            batch_size: int,
            num_epochs: int,
            use_one_hot: bool,
            save_path: str,
            metric_names=(),
            optim_kwargs=None,
            train_writer=None,
            val_writer=None,
    ):
        """
        Creates a default Cropmap segmentation model Trainer, for regular TimeSeries or Image models.
        @param model: Segmentation model to be trained
        @param dataset: CropDataset instance
        @param loss_fn: PyTorch module that will compute loss (this should be a BatchCriterion)
        @param optim_class: PyTorch optimizer that updates model params
        @param num_shots: Number of batched samples from training set to feed to model
        @param batch_size: Batch size of input images for training
        @param num_epochs: Number of epochs to run training
        @param use_one_hot: Whether the mask will use one-hot encoding or class id per pixel
        @param save_path: Path where model weights will be saved
        @param metric_names: Names of metrics that will measure training performance per epoch
        @param optim_kwargs: Keyword arguments for PyTorch optimizer
        @param train_writer: Tensorboard writer for training metrics
        @param val_writer: Tensorboard writer for validation metrics
        """
        super().__init__(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            optim_class=optim_class,
            num_shots=num_shots,
            batch_size=batch_size,
            num_epochs=num_epochs,
            use_one_hot=use_one_hot,
            save_path=save_path,
            metric_names=metric_names,
            optim_kwargs=optim_kwargs,
            train_writer=train_writer,
            val_writer=val_writer
        )

        self.hflip = data_transforms.HorizontalFlipSimCLRTransform()
        self.vflip = data_transforms.VerticalFlipSimCLRTransform()
        self.rot = data_transforms.RotationSimCLRTransform()
        self.crop = data_transforms.RandomResizedCropSimCLRTransform()

    def create_metrics_dict(self, **metrics):
        """
        Creates a metrics dictionary, mapping `metric_name`--> `metric_val`
        The metrics are 1) averaged over all classes 2) per class
        @param loss: Either `None`, or if specified, PyTorch loss number for the epoch
        @param metrics: Each metric should be a numpy array of shape (num_classes)
        @return: {metric_name: metric_val}
        """
        # Dictionary of `class_name` --> index in metric array
        classes = self.dataset.remapped_classes

        metrics_dict = {}

        # First log mean metrics
        for metric_name, metric_arr in metrics.items():
            metric = metric_arr
            metrics_dict[f"mean/{metric_name}"] = np.nanmean(metric)

            # Break down metric by class
            if isinstance(metric_arr, np.ndarray):
                for class_name, i in classes.items():
                    class_metric_name = f'class_{class_name}/{metric_name}'
                    class_metric = metric[i]
                    metrics_dict[class_metric_name] = class_metric

        return metrics_dict

    def format_and_compute_loss(self, features_1, features_2):
        """
        Wrapper function for formatting preds and targets for loss.
        @param features_1: (b, #f, h, w) shape tensor of model outputs, #f is num_features
        @param features_2: (b, #f, h, w) as above different view
        @return: loss value
        """
        return self.loss_fn(features_1, features_2)

    def train_one_step(self, images_1, images_2):
        """
        Performs one training step over a batch.
        Passes the batch of images through the model, and backprops the gradients.
        @param images_1: Tensor of shape (b, c, h, w)
        @param images_2: Tensor of shape (b, c, h, w)
        @return: loss value
        """
        # Flush the gradient buffers
        self.optimizer.zero_grad()

        # Feed model
        # Doing concatentation of lists samples in timeseries here b/c transforms are written for block images
        # images_1 is a list and the concatentation is as in the forward function of simplenet
        # images_2 is already a tensor
        if isinstance(images_1, list):
            images_1 = torch.cat(images_1, dim=1)
            images_2 = torch.cat(images_2, dim=1)
            # images_2 = torch.cat(list(images_2), dim=1)
        n = images_1.shape[0]

        do_hflips = [[ (random()>0.5) for _ in range(n)] for _ in range(2)]
        do_vflips = [[ (random()>0.5) for _ in range(n)] for _ in range(2)]
        do_rots = [[randint(0,3) for _ in range(n)] for _ in range(2)]
        crop_seeds = np.random.uniform(size=(n, 3))

        for i, images in enumerate([images_1, images_2]):
            if i==0: images = self.crop(images, crop_seeds)
            images = self.hflip(images, do_hflips[i])
            images = self.vflip(images, do_hflips[i])
            images = self.rot(images, do_rots[i])
            if i==0:
                images_1 = images
            elif i==1:
                images_2 = images

        features_1 = self.model(images_1)
        features_2 = self.model(images_2)

        for i, features in enumerate([features_1, features_2]):
            features = self.rot(features, do_rots[i], inverse=True)
            features = self.vflip(features, do_hflips[i])
            features = self.hflip(features, do_hflips[i])
            if i==0:
                features_1 = features
            elif i==1:
                features_2 = features
        features_2 = self.crop(features_2, crop_seeds)

        loss = self.format_and_compute_loss(features_1, features_2)

        # Backpropagate
        loss.backward()
        self.optimizer.step()

        return loss

    def val_one_step(self, images_1, images_2):
        """
        Performs one validation step over a batch.
        Passes the batch of images through the model.
        @param images_1: Tensor of shape (b, c, h, w)
        @param images_2: Tensor of shape (b, c, h, w)
        @return: loss value
        """
        # Feed model
        if isinstance(images_1, list):
            images_1 = torch.cat(images_1, dim=1)
            images_2 = torch.cat(images_2, dim=1)
            # images_2 = torch.cat(list(images_2), dim=1)
        n = images_1.shape[0]
        do_hflips = [[ (random()>0.5) for _ in range(n)] for _ in range(2)]
        do_vflips = [[ (random()>0.5) for _ in range(n)] for _ in range(2)]
        do_rots = [[randint(0,3) for _ in range(n)] for _ in range(2)]
        crop_seeds = np.random.uniform(size=(n, 3))

        for i, images in enumerate([images_1, images_2]):
            if i==0: images = self.crop(images, crop_seeds)
            images = self.hflip(images, do_hflips[i])
            images = self.vflip(images, do_vflips[i])
            images = self.rot(images, do_rots[i])
            if i==0:
                images_1 = images
            elif i==1:
                images_2 = images

        features_1 = self.model(images_1)
        features_2 = self.model(images_2)
        
        for i, features in enumerate([features_1, features_2]):
            features = self.rot(features, do_rots[i], inverse=True)
            features = self.vflip(features, do_vflips[i])
            features = self.hflip(features, do_hflips[i])
            if i==0:
                features_1 = features
            elif i==1:
                features_2 = features
        features_2 = self.crop(features_2, crop_seeds)
        loss = self.format_and_compute_loss(features_1, features_2)
        return loss

    def _run_one_epoch(self, loaders, is_train):
        """
        Runs a train/validation loop over all data in dataset
        @param loaders: The train/val loaders
        @param is_train: Whether to train or validate the model
        @return: Dict of {metric_name: metric_val}
        """
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # Set up metrics
        # Note that this is currently overwriting
        # If I don't define metrics in config file then it assumes I want IoU/precision/recall
        # Adding 'loss' to metrics in config is what I want, but default behavior is to automatically log that
        # Should think about cleaner solution but for now overwriting is OK and maintains rest of code
        epoch_metrics = {name: MeanMetric() for name in self.metric_names}
        epoch_metrics["loss"] = MeanMetric()

        for batch_index, (input_1, input_2) in enumerate(loaders):
            # Shift to correct device
            input_1, input_2 = self.dataset.shift_sample_to_device((input_1, input_2), self.device)

            # Input into the model
            feed_func = self.train_one_step if is_train else self.val_one_step
            loss = feed_func(input_1, input_2)
            _metrics = {}
            # Not doing normal calculate_metrics(preds_arr, y_arr, pred_threshold=0)
            # b/c don't have classification
            _metrics["loss"] = loss.item()

            # Update metrics for epoch
            for metric_name, current_val in epoch_metrics.items():
                current_val.update(_metrics[metric_name])

            if self.num_shots is not None and (batch_index+1) == self.num_shots:
                break

        # Get rid of the mean metrics
        epoch_metrics = {metric_name: val.item() for metric_name, val in epoch_metrics.items()}

        # Create metrics dict
        return self.create_metrics_dict(**epoch_metrics)

    def validate_one_epoch(self, val_loaders):
        return self._run_one_epoch(val_loaders, is_train=False)

    def train_one_epoch(self, train_loaders):
        return self._run_one_epoch(train_loaders, is_train=True)

    def train(self, start_epoch):
        """
        Special train func because not saving on IoU, might want to split back out into self-sup trainer
        if we do a lot more besides simclr

        Trains the model for the full number of epochs, from start_epoch to start + num_epochs
        Creates the data loaders from the dataset
        Logs training and validation metrics per epoch
        Saves model if mean val IoU is best seen, or if IoU hasn't dropped too much for 10 epochs
        @param start_epoch: Starting epoch to begin training
        @return:
        """
        # Get the data loaders
        train_loaders, val_loaders, _ = \
            self.dataset.create_data_loaders(batch_size=self.batch_size)
        
        print(val_loaders, len(val_loaders))
        if len(val_loaders)==0:
            train_only = True

        # Variables to keep track of when to checkpoint
        best_loss = np.inf
        epochs_since_last_save = 0
        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            print(f"Starting epoch {epoch + 1}:")

            train_metrics = self.train_one_epoch(train_loaders)
            self.log_metrics(train_metrics, epoch=epoch, phase="train")

            if not train_only:
                val_metrics = self.validate_one_epoch(val_loaders)
                self.log_metrics(val_metrics, epoch=epoch, phase="val")

                # Save model checkpoint if val iou better than best recorded so far.
                loss = val_metrics['mean/loss']
                diff = best_loss - loss
            else:
                print("Skipping val because in train_only mode")
                diff = 1
                loss = None
            if diff > 0 or (epochs_since_last_save > 10 and abs(diff / best_loss) < 0.05):
                best_loss = loss if diff > 0 else best_loss
                epochs_since_last_save = 0

                save_model(self.model, self.save_path)
            else:
                epochs_since_last_save += 1

            print(f"Finished epoch {epoch + 1}.\n")

