import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trainers.trainer import Trainer
from trainers.utils import apply_to_model_parameters

from data_loaders.task_loader import TaskDataset
from metrics import calculate_metrics, MeanMetric


class MAMLTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            dataset: TaskDataset,
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
        self.inner_loop_lr = 0.001
        self.use_higher_order = True

    @staticmethod
    def create_dataset(classifier_name, data_path, data_map_path,
                       classes, interest_classes, transforms):
        return TaskDataset(
            data_path=data_path,
            classes=classes,
            interest_classes=interest_classes,
            data_map_path=data_map_path,
            transforms=transforms,
            train_val_test=(0.8, 0.1, 0.1),
            inf_mode=False
        )

    @staticmethod
    def clone_model(model: nn.Module):
        """
        Recursively clones a nn.Module model such that the clone's parameters are linked
        to the original in the computational graph. We need to do this for the inner loop.
        Inspiration: https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py#L51
        @param [nn.Module] model: PyTorch model to be cloned
        @return: cloned model with gradient links preserved.
        """
        # Inspired from (link broken on two lines):
        # (https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/
        # torch/nn/modules/module.py#L1171)
        def copy_attributes(module):
            replica = module.__new__(type(module))
            replica.__dict__ = module.__dict__.copy()
            replica._parameters = replica._parameters.copy()
            replica._buffers = replica._buffers.copy()
            replica._modules = replica._modules.copy()
            return replica

        cloned_model = apply_to_model_parameters(
            model,
            param_func=lambda p: p.clone(),
            module_func=copy_attributes,
            memo_key_func=lambda p: p.data_ptr,
        )

        return cloned_model

    @staticmethod
    def update_model(model, grads=None, lr=0.001):
        """
        Updates model parameters using data in the .update field of each parameter.
        Specifically, new_param <- param + param.update.
        If grads and learning rate are specified, does SGD update: new_p <- p - lr * grad
        @param model: PyTorch model to be updated
        @param grads: Optional generator of model gradients, should match with model params.
        @param lr: Learning rate for SGD update.
        @return: Updated model with changed parameters.
        """
        if grads is not None:
            for param, grad in zip(model.parameters(), grads):
                param.update = -lr * grad

        updated_model = apply_to_model_parameters(
            model,
            param_func=lambda p: p + p.update,
        )
        return updated_model

    def create_metrics_dict(self, metrics):
        """
        Creates a metrics dictionary, mapping `metric_name`--> `metric_val`
        The metrics are 1) averaged over all classes 2) per class
        @param metrics: Each metric should be a numpy array of shape (num_classes) or scalar
        @return: {metric_name: metric_val}
        """
        # Dictionary of `class_name` --> index in metric array
        classes = self.dataset.remapped_classes

        # TODO: Document this nonsense
        metrics_dict = {}
        for task_name, task_metrics in metrics.items():
            for task_type, epoch_metrics in task_metrics.items():
                for metric_name, metric in epoch_metrics.items():
                    metric_suffix = f'{task_name}/{task_type}/{metric_name}'
                    metrics_dict[f"mean/{metric_suffix}"] = np.nanmean(metric)

                    # Break down metric by class
                    if isinstance(metric, np.ndarray):
                        for class_name, i in classes.items():
                            class_metric_name = f'class_{class_name}/{metric_suffix}'
                            class_metric = metric[i]
                            metrics_dict[class_metric_name] = class_metric

        return metrics_dict

    def validate_one_epoch(self, val_loaders):
        self.model.eval()

        epoch_metrics = {}
        for task_name, val_loader in val_loaders.items():

            # Set up metrics
            task_metrics = {name: MeanMetric() for name in self.metric_names}
            task_metrics["loss"] = MeanMetric()

            for batch_index, (input_t, y) in enumerate(val_loader):
                input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

                # Just evaluate model
                preds = self.model(input_t)
                loss = self.loss_fn(preds, y)

                preds_arr = preds.detach().cpu().numpy()
                y_arr = y.detach().cpu().numpy()
                _metrics = calculate_metrics(preds_arr, y_arr, pred_threshold=0)
                _metrics["loss"] = loss.item()

                # Update metrics for task
                for metric_name, current_val in task_metrics.items():
                    current_val.update(_metrics[metric_name])

            # Get rid of the mean metrics
            epoch_metrics[task_name] = {
                "query": {metric_name: val.item() for metric_name, val in task_metrics.items()}
            }

        return self.create_metrics_dict(epoch_metrics)

    def _feed_support(self, phi, images, labels):
        """
        The phi model weights are updated based on performance on single sample of support data.
        @param phi: Copied model weights from theta at the beginning of task
        @param images: input images shape (b, 9, h, w)
        @param labels: input labels shape (b, #c, h, w)
        @return: predictions, loss
        """
        preds = phi(images)
        loss = self.loss_fn(preds, labels)

        # Manually calculate gradients and update (no optimizer)
        grads = torch.autograd.grad(loss, phi.parameters(), create_graph=self.use_higher_order)

        phi = self.update_model(phi, grads=grads, lr=self.inner_loop_lr)

        return preds, loss

    def _feed_query(self, phi, images, labels):
        """
        The phi model weights are tested on a single sample of query data,
        and the loss is backpropagated to the theta weights, but there is no update performed.
        @param phi: Copied model weights from theta at the beginning of task
        @param images: input images shape (b, 9, h, w)
        @param labels: input labels shape (b, #c, h, w)
        @return: predictions, loss
        """
        preds = phi(images)
        loss = self.loss_fn(preds, labels)

        # Just backpropagate, don't update
        loss.backward(create_graph=self.use_higher_order)

        return loss, preds

    def inner_loop(self, phi, loader, is_support):
        """
        Runs the support/query inner loop of MAML over the entire support/query set.
        @param phi: Copied model weights from theta at the beginning of task.
        @param loader: Either support/query data loader.
        @param is_support: Whether to use support/query
        @return: {metric_name: metric_val} for either support/query set
        """
        feed_func = self._feed_support if is_support else self._feed_query

        # Set up metrics
        task_metrics = {name: MeanMetric() for name in self.metric_names}
        task_metrics["loss"] = MeanMetric()

        for batch_index, (x_input, y) in enumerate(loader):
            x_input, y = self.dataset.shift_sample_to_device((x_input, y), self.device)

            # Run input through phi model weights
            preds, loss = feed_func(phi, x_input, y)

            # Convert to numpy and calculate metrics
            preds_arr = preds.detach().cpu().numpy()
            y_arr = y.detach().cpu().numpy()
            _metrics = calculate_metrics(preds_arr, y_arr, pred_threshold=0)
            _metrics["loss"] = loss.item()

            # Update metrics for task
            for metric_name, current_val in task_metrics.items():
                current_val.update(_metrics[metric_name])

        # Get rid of the mean metrics
        task_metrics = {metric_name: val.item() for metric_name, val in task_metrics.items()}
        return task_metrics

    def train_one_epoch(self, train_loaders):
        self.model.train()

        # Set up epoch metrics per task --> support/query --> {metric_name: metric_val}
        epoch_metrics = {}

        # Zero the gradients before iterating through all tasks
        self.optimizer.zero_grad()

        for task_name, (support_loader, query_loader) in train_loaders.items():

            # Create clone of model for this task
            phi = self.clone_model(self.model)

            # Inner loop
            support_metrics = self.inner_loop(phi, support_loader, is_support=True)
            query_metrics = self.inner_loop(phi, query_loader, is_support=True)

            # Keep metrics ready
            epoch_metrics[task_name] = {"support": support_metrics, "query": query_metrics}

        # Update the outer model theta
        self.optimizer.step()

        return self.create_metrics_dict(epoch_metrics)