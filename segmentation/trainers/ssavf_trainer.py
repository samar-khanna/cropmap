import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from metrics import MeanMetric
from trainers.base_trainer import Trainer
from data_loaders.dataset import CropDataset


class SSAVFTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            loss_fn: nn.Module,
            optim_class: optim.Optimizer,
            num_shots: Optional[int],
            batch_size: int,
            num_epochs: int,
            save_path: str,
            num_display=8,
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
        @param save_path: Path where model weights will be saved
        @param num_display: Number of model preds to display. Grid has 2x due to ground truths
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
            save_path=save_path,
            num_display=num_display,
            metric_names=metric_names,
            optim_kwargs=optim_kwargs,
            train_writer=train_writer,
            val_writer=val_writer
        )
        self.metric_names = []

    @staticmethod
    def create_loss(loss_name, loss_kwargs) -> nn.Module:
        mse = nn.MSELoss(reduction='mean')
        return mse

    def init_checkpoint_metric(self):
        """
        Returns an initial value for the metric used to determine when to save
        model weights (i.e. when to checkpoint)
        @return:
        """
        return np.inf

    def check_checkpoint_metric(self, val_metrics, epochs_since_last_save, prev_best):
        """
        Returns a new value for the metric used to keep track of when to save/checkpoint
        model weights, or None if weights should not be saved
        @param val_metrics: Dictionary of validation metrics for the current epoch
        @param epochs_since_last_save: The number of epochs passed since the last checkpoint
        @param prev_best: The previous best value of the checkpoint metric
        @return:
        """
        theta_loss = val_metrics['theta_loss']
        diff = prev_best - theta_loss
        if diff > 0 or (epochs_since_last_save > 10 and abs(diff / prev_best) < 0.02):
            return theta_loss if diff > 0 else prev_best

        return None

    def format_and_compute_loss(self, preds, targets):
        """
        Wrapper function for formatting preds and targets for loss.
        @param preds: (b, #c, h, w) shape tensor of model outputs, #c is num classes
        @param targets: (b, 1 or #c, h, w) shape tensor of targets.
                        If only 1 channel, then squeeze it for CrossEntropy
        @return: loss value
        """
        loss = self.loss_fn(preds, targets)
        return loss

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
        x_aligned, x_shift, theta_pred, theta_shift = self.model(images)
        loss = self.format_and_compute_loss(theta_pred, theta_shift)

        # Backpropagate
        loss.backward()
        self.optimizer.step()

        return x_aligned, x_shift, loss

    def val_one_step(self, images, labels):
        """
        Performs one validation step over a batch.
        Passes the batch of images through the model.
        @param images: Tensor of shape (b, c, h, w)
        @param labels: Tensor of shape (b, c, h, w)
        @return: model predictions, loss value
        """
        # Feed model
        x_aligned, x_shift, theta_pred, theta_shift = self.model(images)
        loss = self.format_and_compute_loss(theta_pred, theta_shift)

        return x_aligned, x_shift, loss

    def _run_one_epoch(self, loaders, is_train):
        """
        Runs a train/validation loop over all data in dataset
        @param loaders: The train/val loaders
        @param is_train: Whether to train or validate the model
        @return: Dict of {metric_name: metric_val}
        """
        # Set up metrics
        epoch_metrics = {name: MeanMetric() for name in self.metric_names}
        epoch_metrics["theta_loss"] = MeanMetric()
        epoch_metrics["euc_loss"] = MeanMetric()

        for batch_index, (input_t, y) in enumerate(loaders):
            # Shift to correct device
            input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

            # Input into the model
            feed_func = self.train_one_step if is_train else self.val_one_step
            x_aligned, x_shift, loss = feed_func(input_t, y)
            euc_loss = self.loss_fn(x_aligned, x_shift)

            # TODO: We can technically get per class euclidean loss for x_align x_shift
            _metrics = {"theta_loss": loss.item(), "euc_loss": euc_loss.item()}

            # Update metrics for epoch
            for metric_name, current_val in epoch_metrics.items():
                current_val.update(_metrics[metric_name])

            if self.num_shots is not None and (batch_index + 1) == self.num_shots:
                break

        # Get rid of the mean metrics
        epoch_metrics = {metric_name: val.item() for metric_name, val in epoch_metrics.items()}
        # metrics_dict = self.create_metrics_dict(**epoch_metrics)

        # Create metrics dict
        return epoch_metrics, None

    def validate_one_epoch(self, val_loaders):
        self.model.eval()
        with torch.no_grad():
            return self._run_one_epoch(val_loaders, is_train=False)

    def train_one_epoch(self, train_loaders):
        self.model.train()
        return self._run_one_epoch(train_loaders, is_train=True)
