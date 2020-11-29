import torch.nn as nn
import torch.optim as optim

from trainers.trainer import Trainer
from data_loaders.dataset import CropDataset
from metrics import create_metrics_dict, calculate_metrics, MeanMetric


class DefaultTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            loss_fn: nn.Module,
            optim_class: optim.Optimizer,
            batch_size: int,
            num_epochs: int,
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
        @param loss_fn: PyTorch module that will compute loss
        @param optim_class: PyTorch optimizer that updates model params
        @param batch_size: Batch size of input images for training
        @param num_epochs: Number of epochs to run training
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
            batch_size=batch_size,
            num_epochs=num_epochs,
            save_path=save_path,
            metric_names=metric_names,
            optim_kwargs=optim_kwargs,
            train_writer=train_writer,
            val_writer=val_writer
        )

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
        loss = self.loss_fn(preds, labels)

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
        loss = self.loss_fn(preds, labels)

        return preds, loss

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
            _metrics = calculate_metrics(preds_arr, y_arr, pred_threshold=0)
            _metrics["loss"] = loss.item()

            # Update metrics for epoch
            for metric_name, current_val in epoch_metrics.items():
                current_val.update(_metrics[metric_name])

        # Get rid of the mean metrics
        epoch_metrics = {metric_name: val.item() for metric_name, val in epoch_metrics.items()}

        # Create metrics dict
        return create_metrics_dict(self.dataset.remapped_classes, **epoch_metrics)

    def validate_one_epoch(self, val_loaders):
        return self._run_one_epoch(val_loaders, is_train=False)

    def train_one_epoch(self, train_loaders):
        return self._run_one_epoch(train_loaders, is_train=True)
