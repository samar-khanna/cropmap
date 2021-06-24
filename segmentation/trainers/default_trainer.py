import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from trainers.base_trainer import Trainer
from data_loaders.dataset import CropDataset
from trainers.trainer_utils import compute_masked_loss
from metrics import calculate_metrics, MeanMetric, mean_accuracy_from_images


class DefaultTrainer(Trainer):
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
        Creates a default segmentation model Trainer, for regular TimeSeries or Image models.
        @param model: Segmentation model to be trained
        @param dataset: CropDataset instance
        @param loss_fn: PyTorch module that will compute loss
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

    def init_checkpoint_metric(self):
        """
        Returns -∞ value for initial best validation IoU
        @return:
        """
        return -np.inf

    def check_checkpoint_metric(self, val_metrics, epochs_since_last_save, prev_best):
        """
        Checks if class mean validation IoU for current epoch is better than previous high.
        Returns updated IoU if epoch val IoU is better, or if >10 epochs have passed and
        current epoch val IoU is less than 5% smaller than previous best.
        @param val_metrics:
        @param epochs_since_last_save:
        @param prev_best:
        @return: Updated best validation IoU, or None if shouldn't checkpoint weights.
        """
        val_iou = val_metrics['mean/iou']
        diff = val_iou - prev_best
        if diff > 0 or (epochs_since_last_save > 10 and abs(diff / prev_best) < 0.05):
            return val_iou if diff > 0 else prev_best

        return None

    def format_and_compute_loss(self, preds, targets):
        """
        Wrapper function for formatting preds and targets for loss.
        @param preds: (b, #c, h, w) shape tensor of model outputs, #c is num classes
        @param targets: (b, 1 or #c, h, w) shape tensor of targets.
                        If only 1 channel, then squeeze it for CrossEntropy
        @return: loss value
        """
        # If there is no channel dimension in the target, remove it for CrossEntropy
        if not self.use_one_hot:
            targets = targets.squeeze(1).type(torch.long)

        return compute_masked_loss(self.loss_fn, preds, targets, invalid_value=-1)

    def train_one_step(self, images, labels):
        """
        Performs one training step over a batch.
        Passes the batch of images through the model, and backprops the gradients.
        @param images: Tensor of shape (b, c, h, w)
        @param labels: Tensor of shape (b, c, h, w)
        @return: model predictions, loss value
        """
        # Flush the gradient buffers
        self.optimizer.zero_grad()

        # Feed model
        preds = self.model(images)
        loss = self.format_and_compute_loss(preds, labels)

        # Backpropagate
        loss.backward()
        self.optimizer.step()

        return preds, loss

    def val_one_step(self, images, labels):
        """
        Performs one validation step over a batch.
        Passes the batch of images through the model.
        @param images: Tensor of shape (b, c, h, w)
        @param labels: Tensor of shape (b, c, h, w)
        @return: model predictions, loss value
        """
        # Feed model
        preds = self.model(images)
        loss = self.format_and_compute_loss(preds, labels)

        return preds, loss

    def _run_one_epoch(self, loaders, is_train):
        """
        Runs a train/validation loop over all data in dataset
        @param loaders: The train/val loaders
        @param is_train: Whether to train or validate the model
        @return: Dict of {metric_name: metric_val}
        """
        # Set up metrics
        epoch_metrics = {name: MeanMetric() for name in self.metric_names}
        epoch_metrics["loss"] = MeanMetric()

        for batch_index, (input_t, y) in enumerate(loaders):
            # Shift to correct device
            input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

            # Input into the model
            feed_func = self.train_one_step if is_train else self.val_one_step
            preds, loss = feed_func(input_t, y)

            # Convert to numpy and calculate metrics
            preds_arr = preds.detach().cpu().numpy()
            y_arr = y.detach().cpu().numpy()
            if not self.use_one_hot:
                y_arr = self.dataset.one_hot_mask(y_arr, self.dataset.num_classes)

            _metrics = calculate_metrics(preds_arr, y_arr, pred_threshold=0)
            _metrics["loss"] = loss.item()
            _metrics["accuracy"] = mean_accuracy_from_images(preds_arr, y_arr, pred_threshold=0)

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
        self.model.eval()
        with torch.no_grad():
            return self._run_one_epoch(val_loaders, is_train=False)

    def train_one_epoch(self, train_loaders):
        self.model.train()
        return self._run_one_epoch(train_loaders, is_train=True)
