import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from utils.colors import get_cmap
from trainers.base_trainer import Trainer
from data_loaders.dataset import CropDataset
from trainers.trainer_utils import compute_masked_loss, get_display_indices
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
        @param use_one_hot: Whether the mask will use one-hot encoding or class id per pixel
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
            use_one_hot=use_one_hot,
            save_path=save_path,
            num_display=num_display,
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

            # Break down metric by class if metric is an array
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

    def _get_display_indices(self, batch_index, len_loader, curr_len_display):
        return get_display_indices(
            batch_index, self.batch_size, self.num_display, len_loader, curr_len_display
        )

    def _format_for_display(self, pred_batch, gt_batch, idx=0):
        # TODO: Put this function in CropDataset
        # get first in batch as convention
        pred, gt = pred_batch[idx], gt_batch[idx]  # (b, c, h, w) -> (c, h, w)
        if self.use_one_hot:
            gt = self.dataset.inverse_one_hot_mask(gt)  # (c, h, w) -> (1, h, w)
        gt = np.squeeze(gt)  # (1, h, w) -> (h, w)
        unk_mask = gt == -1  # (h, w)

        pred = np.argmax(pred, axis=0)  # (c, h, w) -> (h, w)

        # Map to original class idx
        pred = self.dataset.map_idx_to_class[pred]  # (h, w)
        pred[unk_mask] = -1
        gt = self.dataset.map_idx_to_class[gt]  # (h, w)
        gt[unk_mask] = -1

        # Colorise the images and drop alpha channel
        cmap = get_cmap(self.dataset.all_classes)
        color_gt = cmap(gt).transpose(2, 0, 1)  # (h,w) -> (h,w,4) -> (4,h,w)
        color_pred = cmap(pred).transpose(2, 0, 1)  # (h,w) -> (h,w,4) -> (4,h,w)

        # Drop alpha channel
        display_im = np.stack((color_pred[:3, ...], color_gt[:3, ...]), axis=0)
        return display_im  # (2, 3, h, w)

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
        epoch_metrics["accuracy"] = MeanMetric()

        # Set up batch of images to display
        display_batch = []

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

            # Check and add pred/gt to running display image batch
            for idx in self._get_display_indices(batch_index, len(loaders), len(display_batch)):
                display_batch.append(self._format_for_display(preds_arr, y_arr, idx))

            if self.num_shots is not None and (batch_index + 1) == self.num_shots:
                break

        # Get rid of the mean metrics
        epoch_metrics = {metric_name: val.item() for metric_name, val in epoch_metrics.items()}
        metrics_dict = self.create_metrics_dict(**epoch_metrics)

        # Concatenate the images along the batch dimension
        display_batch = np.concatenate(display_batch) if len(display_batch) > 0 else None

        # Create metrics dict
        return metrics_dict, display_batch

    def validate_one_epoch(self, val_loaders):
        self.model.eval()
        with torch.no_grad():
            return self._run_one_epoch(val_loaders, is_train=False)

    def train_one_epoch(self, train_loaders):
        self.model.train()
        return self._run_one_epoch(train_loaders, is_train=True)
