import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from trainers.default_trainer import DefaultTrainer
from data_loaders.dataset import CropDataset
from trainers.trainer_utils import compute_masked_loss
from metrics import calculate_metrics, MeanMetric


class MissingMonthTrainer(DefaultTrainer):
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

    def create_metrics_dict(self, num_classes, **metrics):
        """
        Creates a metrics dictionary, mapping `metric_name`--> `metric_val`
        The metrics are 1) averaged over all classes 2) per class
        @param num_classes: Number of classes in final output / label
        @param metrics: Each metric should be a numpy array of shape (num_classes)
        @return: {metric_name: metric_val}
        """
        # Dictionary of `class_name` --> index in metric array
        # Dictionary of `class_name` --> index in metric array
        classes = list(range(num_classes))

        metrics_dict = {}

        # First log mean metrics
        for metric_name, metric_arr in metrics.items():
            metric = metric_arr
            metrics_dict[f"mean/{metric_name}"] = np.nanmean(metric)

            # Break down metric by class
            if isinstance(metric_arr, np.ndarray):
                for class_i in classes:
                    class_metric_name = f'month_{class_i}/{metric_name}'
                    class_metric = metric[class_i]
                    metrics_dict[class_metric_name] = class_metric

        return metrics_dict

    def format_and_compute_loss(self, preds, targets):
        """
        Wrapper function for formatting preds and targets for loss.
        @param preds: (b, #c, h, w) shape tensor of model outputs, #c is num classes
        @param targets: (b, 1 or #c, h, w) shape tensor of targets.
                        If only 1 channel, then squeeze it for CrossEntropy
        @return: loss value
        """
        # TODO
        # If there is no channel dimension in the target, remove it for CrossEntropy
        if not self.dataset.use_one_hot:
            targets = targets.squeeze(1).type(torch.long)

        return compute_masked_loss(self.loss_fn, preds, targets, invalid_value=-1)

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
            # As in rotation pretext want to have all frames classified
            n_months = len(input_t)
            b, _, h, w = y.shape

            # Get list of repeated one-hots for each month then stack
            drop_month = torch.randint(0, n_months, size=(b,))
            one_hot_mat_list = [
                torch.eye(n_months)[d].repeat(h, w, 1).permute(2, 0, 1)  # (t, h, w)
                for d in drop_month
            ]
            y = torch.stack(one_hot_mat_list, dim=0)  # (b, t, h, w)

            batch_range = torch.arange(b).unsqueeze(-1)  # (b, 1)
            keep_months = torch.as_tensor(
                [[j for j in range(n_months) if j != drop_month[i]] for i in range(b)]
            )  # (b, t-1)

            # Drop month from input
            # ASSUME: Equal time series length for all images in batch
            # Change from (t, b, c, h, w) -> (b, t, c, h, w)
            batch_first_input = torch.stack(input_t, dim=0).transpose(0, 1)
            batch_first_input = batch_first_input[batch_range, keep_months, :]  # (b, t-1, c,h,w)
            input_t = batch_first_input.transpose(0, 1)  # (t-1, b, c,h,w)
            input_t = [t for t in input_t]  # back to list for correct format

            # Shift to correct device
            input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

            # Input into the model
            feed_func = self.train_one_step if is_train else self.val_one_step
            preds, loss = feed_func(input_t, y)

            # Convert to numpy and calculate metrics
            preds_arr = preds.detach().cpu().numpy()
            y_arr = y.detach().cpu().numpy()
            if not self.dataset.use_one_hot:
                y_arr = self.dataset.one_hot_mask(y_arr, n_months)

            _metrics = calculate_metrics(preds_arr, y_arr, pred_threshold=0)
            _metrics["loss"] = loss.item()

            # Update metrics for epoch
            for metric_name, current_val in epoch_metrics.items():
                current_val.update(_metrics[metric_name])

            if self.num_shots is not None and (batch_index+1) == self.num_shots:
                break

        # Get rid of the mean metrics
        epoch_metrics = {metric_name: val.item() for metric_name, val in epoch_metrics.items()}

        # Create metrics dict
        return self.create_metrics_dict(n_months, **epoch_metrics), None
