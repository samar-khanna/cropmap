import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

import models.loss as custom_loss
from data_loaders.dataset import CropDataset
from utils.utils import create_dirs
from utils.loading import load_model, save_model, create_dataset


class Trainer:
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
            num_display=8,
            metric_names=(),
            optim_kwargs=None,
            train_writer=None,
            val_writer=None,
    ):
        """
        Base class for all different Trainers. Creates a basic, abstract training loop.
        @param model: Segmentation model to be trained
        @param dataset: CropDataset instance
        @param loss_fn: PyTorch module that will compute loss
        @param optim_class: PyTorch optimizer that updates model params
        @param num_shots: Number of batched samples from training set to feed to model
        @param batch_size: Batch size of input images for training
        @param num_epochs: Number of epochs to run training
        @param use_one_hot: Whether the mask will use one-hot encoding or class id per pixel
        @param save_path: Path where model weights will be saved
        @param num_display: Number of model preds to display. Grid has 2x due to ground truths
        @param metric_names: Names of metrics that will measure training performance per epoch
        @param optim_kwargs: Keyword arguments for PyTorch optimizer
        @param train_writer: Tensorboard writer for training metrics
        @param val_writer: Tensorboard writer for validation metrics
        """
        self.num_shots = num_shots
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_display = num_display

        # Get list of metrics to use for training
        self.metric_names = metric_names
        if len(self.metric_names) == 0:
            self.metric_names = ["iou", "prec", "recall", "class_acc"]

        # Set up loggers
        self.train_writer = train_writer
        self.val_writer = val_writer

        # Set up dataset
        self.dataset = dataset

        # Set up available device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Get model
        self.save_path = save_path
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        self.model = model
        self.model.to(self.device)

        # Set up loss
        self.use_one_hot = use_one_hot
        self.loss_fn = loss_fn
        self.loss_fn.to(self.device)

        # Set up optimizer(s)
        weights = filter(lambda w: w.requires_grad, self.model.parameters())
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        self.optimizer = optim_class(weights, **optim_kwargs)

    @classmethod
    def create_trainer(
            cls,
            data_path,
            data_map_path,
            out_dir,
            exp_name,
            trainer_config,
            model_config,
            classes,
            checkpoint_path,
            freeze_backbone,
            **kwargs
    ):
        """
        Creates a Trainer out of raw arguments
        @param data_path: Path to directory containing all datasets
        @param data_map_path: Path to .json file containing dataset split information
        @param out_dir: Path to directory where training results will be stored
        @param exp_name: Name of experiment to create results directory (if None, uses configs)
        @param trainer_config: JSON config file containing Trainer parameters
        @param model_config: JSON config file containing Model parameters
        @param classes: JSON file containing Cropmap class name --> class id
        @param checkpoint_path: Path to .bin checkpoint file containing model weights
        @param freeze_backbone: Whether to freeze model backbone while training
        @param kwargs: Extra keyword arguments to pass to init function.
        @return: Initialised Trainer
        """
        # Create output directory, save directory and metrics directories.
        exp_name = exp_name if exp_name is not None else\
            "_".join((model_config["name"], trainer_config["name"]))
        out_dir = os.path.join(out_dir if out_dir is not None else data_path, exp_name)
        save_dir = os.path.join(out_dir, "checkpoints")
        metrics_dir = os.path.join(out_dir, "metrics")
        create_dirs(out_dir, save_dir, metrics_dir)

        save_path = os.path.join(save_dir, f"checkpoint.bin")

        # SAVE config file in output directory at beginning of training
        cls.save_config(trainer_config, out_dir, 'trainer_config')
        cls.save_config(model_config, out_dir, 'model_config')

        # Set up loggers
        train_writer, val_writer = cls.create_loggers(metrics_dir)

        # Set up loss
        loss_name = trainer_config.get("loss", "CrossEntropyLoss")
        loss_kwargs = trainer_config.get("loss_kwargs", {})
        loss_fn = cls.create_loss(loss_name, loss_kwargs)

        # Set up optim class but don't initialise
        optim_name = trainer_config.get("optimizer", "Adam")
        optim_kwargs = trainer_config.get("optimizer_kwargs", {"lr": 0.001})
        optimizer_class = cls.create_optimizer(optim_name)

        # Set up dataset
        use_one_hot = loss_name.lower().find("bce") > -1 or loss_name.lower().find("focal") > -1
        classifier_name = model_config["classifier"].lower()
        interest_classes = trainer_config.get("interest_classes", [])
        transforms = trainer_config.get("transforms", {})
        dataset = cls.create_dataset(classifier_name, data_path, data_map_path,
                                     classes, interest_classes, use_one_hot, transforms)

        # TODO: Find a way to break this link between model and trainer config
        # Set up model using its config file and number of classes from trainer config.
        num_classes = trainer_config.get("num_classes", len(interest_classes or classes.keys()))
        if type(checkpoint_path) is bool and checkpoint_path:
            checkpoint_path = save_path
        model = load_model(model_config, num_classes, checkpoint_path, freeze_backbone)

        return cls(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            optim_class=optimizer_class,
            num_shots=trainer_config.get("num_shots", None),
            batch_size=trainer_config.get("batch_size", 32),
            num_epochs=trainer_config.get("epochs", 200),
            use_one_hot=use_one_hot,
            save_path=save_path,
            num_display=trainer_config.get("num_display", 8),
            metric_names=trainer_config.get("metrics", []),
            optim_kwargs=optim_kwargs,
            train_writer=train_writer,
            val_writer=val_writer,
            **kwargs
        )

    @staticmethod
    def create_loggers(metrics_path):
        """
        Creates train and validation Tensorboards
        @param metrics_path: Path to directory where training metric outputs will be stored.
        @return: train_writer, val_writer
        """
        # Set up tensorboards
        train_writer = SummaryWriter(log_dir=os.path.join(metrics_path, 'train'))
        val_writer = SummaryWriter(log_dir=os.path.join(metrics_path, 'val'))
        logging.basicConfig(filename=os.path.join(metrics_path, "log.log"), level=logging.INFO)

        return train_writer, val_writer

    @staticmethod
    def create_dataset(classifier_name, data_path, data_map_path,
                       classes, interest_classes, use_one_hot, transforms):
        """
        Creates a CropDataset
        @param classifier_name: Name of model classifier, which determines which dataset to use
        @param data_path: Path to directory containing datasets
        @param data_map_path: Path to .json file containing dataset split information
        @param classes: JSON file containing class name --> class id
        @param interest_classes: List of specific classes (subset) to run experiment
        @param use_one_hot: Whether the mask will use one-hot encoding or class id per pixel.
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
            use_one_hot=use_one_hot,
            inf_mode=False,
        )

    @staticmethod
    def create_loss(loss_name, loss_kwargs) -> nn.Module:
        """
        Creates an initialised PyTorch loss nn.Module
        @param loss_name: Name of loss module, either in custom_loss or PyTorch nn
        @param loss_kwargs: Keyword arguments to initialise the loss
        @return: instantiated loss module
        """
        # If weights are given, then convert to tensor and normalize to sum to 1.
        for key in loss_kwargs:
            if key.find("weight") > -1:
                loss_kwargs[key] = \
                    torch.as_tensor(loss_kwargs[key], dtype=torch.float32) / sum(loss_kwargs[key])
        if loss_name in custom_loss.__dict__:
            loss_fn = custom_loss.__dict__[loss_name](**loss_kwargs)
        elif loss_name in nn.__dict__:
            loss_kwargs['reduction'] = 'none'
            loss_fn = nn.__dict__[loss_name](**loss_kwargs)
        else:
            raise ValueError(("Invalid PyTorch loss. The name must exactly match a loss"
                              " in the nn module or a loss defined in models/loss.py"))
        return loss_fn

    @staticmethod
    def create_optimizer(optim_name):
        """
        Instantiates the optimizer based on name.
        Defaults to using Adam(lr=0.0001)
        @param optim_name: Valid optimizer name in torch.optim module
        @return: PyTorch optimizer class
        """
        # Set up optimizer
        assert optim_name in optim.__dict__, \
            "Invalid PyTorch optimizer. Expected exact match with optimizer in the optim module"

        # Only optimize on unfrozen weights.
        optimizer = optim.__dict__[optim_name]

        return optimizer

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

    def log_metrics(self, metrics_dict, epoch, phase):
        """
        Logs metrics to tensorflow summary writers, and to a log file.
        Also prints mean metrics for the epoch
        @param metrics_dict: {metric_name: metric value}
        @param epoch: Current epoch number
        @param phase: One of train/val
        """
        writer = self.train_writer if phase.lower() == "train" else self.val_writer
        # Do it separately for tf writers.
        if writer is not None:
            for metric_name, metric_value in metrics_dict.items():
                writer.add_scalar(metric_name, metric_value, global_step=epoch + 1)

        print(f"Phase: {phase}")
        for metric_name, metric_value in metrics_dict.items():
            logging.info(f"Epoch {epoch + 1}, Phase {phase}, {metric_name}: {metric_value}")

            if not metric_name.startswith('class'):
                print(f"{metric_name}: {metric_value}")

    def log_images(self, im_batch, epoch, phase):
        """
        Adds batch of images to the tensorboard
        @param im_batch: (b, 3, h, w) RGB image tensor/np array.
        @param epoch: Current epoch number
        @param phase: One of train/val
        """
        writer = self.train_writer if phase.lower() == "train" else self.val_writer
        if im_batch is not None and writer is not None:
            writer.add_images(f"{phase}_images", im_batch, epoch + 1, dataformats='NCHW')

    def init_checkpoint_metric(self):
        """
        Returns an initial value for the metric used to determine when to save
        model weights (i.e. when to checkpoint)
        @return:
        """
        raise NotImplementedError("This should be overwritten")

    def check_checkpoint_metric(self, val_metrics, epochs_since_last_save, prev_best):
        """
        Returns a new value for the metric used to keep track of when to save/checkpoint
        model weights, or None if weights should not be saved
        @param val_metrics: Dictionary of validation metrics for the current epoch
        @param epochs_since_last_save: The number of epochs passed since the last checkpoint
        @param prev_best: The previous best value of the checkpoint metric
        @return:
        """
        raise NotImplementedError("This should be overwritten")

    def validate_one_epoch(self, val_loaders):
        """
        Runs validation for one epoch, involving a full pass over validation data
        @param val_loaders: CropDataset dataloader(s) for validation data
        @return: Metrics for the validation epoch {metric_name: metric_val}
        """
        raise NotImplementedError("This should be overwritten")

    def train_one_epoch(self, train_loaders):
        """
        Trains the model for one epoch, involving a full pass over training data
        @param train_loaders: CropDataset dataloader(s) for training data
        @return: Metrics for the training epoch {metric_name: metric_val}
        """
        raise NotImplementedError("This should be overwritten")

    def train(self, start_epoch):
        """
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

        # Variables to keep track of when to checkpoint
        best_val_metric = self.init_checkpoint_metric()
        epochs_since_last_save = 0
        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            print(f"Starting epoch {epoch + 1}:")

            train_metrics, train_display = self.train_one_epoch(train_loaders)
            self.log_metrics(train_metrics, epoch=epoch, phase="train")
            self.log_images(train_display, epoch=epoch, phase='train')

            val_metrics, val_display = self.validate_one_epoch(val_loaders)
            self.log_metrics(val_metrics, epoch=epoch, phase="val")
            self.log_images(val_display, epoch=epoch, phase='val')

            # Save model checkpoint if checkpointing condition is satisfied
            new_best_val_metric = self.check_checkpoint_metric(
                val_metrics, epochs_since_last_save, best_val_metric
            )
            if new_best_val_metric is not None:
                best_val_metric = new_best_val_metric
                epochs_since_last_save = 0

                save_model(self.model, self.save_path)
            else:
                epochs_since_last_save += 1

            print(f"Finished epoch {epoch + 1}.\n")
