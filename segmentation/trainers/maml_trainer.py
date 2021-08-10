import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from trainers.base_trainer import Trainer
from trainers.trainer_utils import apply_to_model_parameters, compute_masked_loss

from data_loaders.task_loader import TaskDataset
from metrics import calculate_metrics, MeanMetric, mean_accuracy_from_images


class MAMLTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            dataset: TaskDataset,
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
            inner_loop_lr=0.001,
            use_higher_order=True,
    ):
        """
        Creates a default Cropmap segmentation model Trainer, for regular TimeSeries or Image models.
        @param model: Segmentation model to be trained
        @param dataset: CropDataset instance
        @param loss_fn: PyTorch module that will compute loss
        @param optim_class: PyTorch optimizer that updates model params
        @param num_shots: Number of batched samples from support/query to feed to model
        @param batch_size: Batch size of input images for training
        @param num_epochs: Number of epochs to run training
        @param save_path: Path where model weights will be saved
        @param num_display: Number of model preds to display. Grid has 2x due to ground truths
        @param metric_names: Names of metrics that will measure training performance per epoch
        @param optim_kwargs: Keyword arguments for PyTorch optimizer
        @param train_writer: Tensorboard writer for training metrics
        @param val_writer: Tensorboard writer for validation metrics
        @param inner_loop_lr: Learning rate for support set SGD update
        @param use_higher_order: Whether to use higher order grad to track support set update.
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
        self.inner_loop_lr = inner_loop_lr
        self.use_higher_order = use_higher_order

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
        maml_kwargs = trainer_config.get("trainer_kwargs", {})
        return super().create_trainer(
            data_path=data_path,
            data_map_path=data_map_path,
            out_dir=out_dir,
            exp_name=exp_name,
            trainer_config=trainer_config,
            model_config=model_config,
            classes=classes,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
            **maml_kwargs
        )

    @staticmethod
    def create_dataset(classifier_name, data_path, data_map_path,
                       classes, interest_classes, use_one_hot, transforms):
        return TaskDataset(
            data_path=data_path,
            classes=classes,
            interest_classes=interest_classes,
            data_map_path=data_map_path,
            use_one_hot=use_one_hot,
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
    def update_model(model, loss=None, lr=0.001, use_higher_order=True):
        """
        Updates model parameters using data in the .update field of each parameter.
        Specifically, new_param <- param + param.update.
        If grads and learning rate are specified, does SGD update: new_p <- p - lr * grad
        @param model: PyTorch model to be updated
        @param loss: Optional loss value which will be used to differentiate model
        @param lr: Learning rate for SGD update.
        @param use_higher_order: If loss is not None, can differentiate with higher order grad
        @return: Updated model with changed parameters.
        """
        if loss is not None:
            # Manually calculate gradients and update (no optimizer)
            grads = torch.autograd.grad(
                loss,
                (p for p in model.parameters() if p.requires_grad),
                create_graph=use_higher_order
            )

            # Regen the generator
            diff_params = (p for p in model.parameters() if p.requires_grad)
            for param, grad in zip(diff_params, grads):
                if grad is not None:
                    param.update = -lr * grad

        # TODO: Figure out why memo_key_func is identity and if p.data_ptr works.
        updated_model = apply_to_model_parameters(
            model,
            param_func=lambda p: p + p.update,
            param_check_func=lambda p: getattr(p, 'update', None) is not None,
            memo_key_func=lambda p: p,
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

        # TODO: Document this nonsense and do something about the aggregate
        aggregate_metrics = {}
        metrics_dict = {}
        for task_name, task_metrics in metrics.items():
            for task_type, epoch_metrics in task_metrics.items():
                for metric_name, metric in epoch_metrics.items():
                    # Record aggregate across tasks
                    aggregate = aggregate_metrics.get(f'mean/{metric_name}', MeanMetric())
                    aggregate.update(np.nanmean(metric))
                    aggregate_metrics[f'mean/{metric_name}'] = aggregate

                    # Fine-grained
                    metric_suffix = f'{task_name}/{task_type}/{metric_name}'
                    metrics_dict[f"mean/{metric_suffix}"] = np.nanmean(metric)

                    # Break down metric by class
                    if isinstance(metric, np.ndarray):
                        for class_name, i in classes.items():
                            class_metric_name = f'class_{class_name}/{metric_suffix}'
                            class_metric = metric[i]
                            metrics_dict[class_metric_name] = class_metric

        # Merge
        return {**metrics_dict, **{k: v.item() for k, v in aggregate_metrics.items()}}

    def init_checkpoint_metric(self):
        """
        Initialises the MAML checkpoint metric to 1 (arbitrary)
        @return:
        """
        return 1

    def check_checkpoint_metric(self, val_metrics, epochs_since_last_save, prev_best):
        """
        Checks if epochs since last save is greater than 10 + noise
        @param val_metrics:
        @param epochs_since_last_save:
        @param prev_best:
        @return:
        """
        # TODO: Update this to include a better method for checkpointing
        if epochs_since_last_save > 10 + np.random.randint(-2, 3):
            return 1
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
        if not self.dataset.use_one_hot:
            targets = targets.squeeze(1).type(torch.long)

        return compute_masked_loss(self.loss_fn, preds, targets, invalid_value=-1)

    def validate_one_epoch(self, val_loaders):
        self.model.eval()

        epoch_metrics = {}
        for task_name, (support_loader, query_loader) in val_loaders.items():

            # Set up metrics
            task_metrics = {name: MeanMetric() for name in self.metric_names}
            task_metrics["loss"] = MeanMetric()
            task_metrics["accuracy"] = MeanMetric()

            for batch_index, (input_t, y) in enumerate(query_loader):
                input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

                # Just evaluate model
                preds = self.model(input_t)
                loss = self.format_and_compute_loss(preds, y)

                preds_arr = preds.detach().cpu().numpy()
                y_arr = y.detach().cpu().numpy()
                if not self.dataset.use_one_hot:
                    y_arr = self.dataset.one_hot_mask(y_arr, self.dataset.num_classes)

                _metrics = calculate_metrics(preds_arr, y_arr, pred_threshold=0)
                _metrics["loss"] = loss.item()
                _metrics["accuracy"] = mean_accuracy_from_images(preds_arr, y_arr, pred_threshold=0)

                # Update metrics for task
                for metric_name, current_val in task_metrics.items():
                    current_val.update(_metrics[metric_name])

            # Get rid of the mean metrics
            epoch_metrics[task_name] = {
                "query": {metric_name: val.item() for metric_name, val in task_metrics.items()}
            }

        return self.create_metrics_dict(epoch_metrics), None

    def _feed_support(self, phi, images, labels):
        """
        The phi model weights are updated based on performance on single sample of support data.
        @param phi: Copied model weights from theta at the beginning of task
        @param images: input images shape (b, 9, h, w)
        @param labels: input labels shape (b, #c, h, w)
        @return: predictions, loss
        """
        preds = phi(images)
        loss = self.format_and_compute_loss(preds, labels)

        # Manually calculate gradients and update (no optimizer)
        phi = self.update_model(phi, loss=loss, lr=self.inner_loop_lr,
                                use_higher_order=self.use_higher_order)

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
        loss = self.format_and_compute_loss(preds, labels)

        # Just backpropagate, don't update. We don't need higher order here.
        # Backward pass will accumulate gradients.
        loss.backward(retain_graph=True)

        return preds, loss

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

            # Only pass few shots to learner
            if self.num_shots is not None and (batch_index+1) == self.num_shots:
                break

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
            query_metrics = self.inner_loop(phi, query_loader, is_support=False)

            # Keep metrics ready
            epoch_metrics[task_name] = {"support": support_metrics, "query": query_metrics}

        # Update the outer model theta
        self.optimizer.step()

        return self.create_metrics_dict(epoch_metrics), None
