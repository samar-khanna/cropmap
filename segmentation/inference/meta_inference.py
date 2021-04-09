import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import Counter

from data_loaders.dataset import CropDataset
from inference.base_inference import InferenceAgent
from data_loaders.task_loader import TaskDataset
from metrics import create_metrics_dict, confusion_matrix, MeanMetric

from trainers.base_trainer import Trainer
from trainers.utils import compute_masked_loss


class MetaInferenceAgent(InferenceAgent):

    _SAVED_TRAINER_CONFIG_NAME = "meta_inference_config"
    _DEFAULT_BATCH_SIZE = 1

    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            out_dir: str,
            exp_name: str,
            loss_fn,
            optim_class,
            optim_kwargs,
            batch_size: int = 1,
            shot_list: tuple = (1, 4, 8, 12, 16, 20, 24),
            reps_per_shot: int = 50,
            num_trials: int = 10,
            metric_names=(),
    ):
        """
        Handles inference, including saving metrics and TODO: images
        @param model: Segmentation model to evaluate
        @param dataset: CropDataset instance
        @param out_dir: Output directory where inference results will be saved
        @param exp_name: Unique name for experiment (used for saving metrics file)
        @param batch_size: Batch size of input images for inference (default 1)
        @param shot_list: Max number of input shots before inference is performed
        @param reps_per_shot: Number of gradient updates made per shot
        @param metric_names: Names of metrics that will measure inference performance
        """
        super().__init__(model, dataset, batch_size, out_dir, exp_name, metric_names)
        self.shot_list = shot_list
        self.reps_per_shot = reps_per_shot
        self.num_trials = num_trials

        # Set up loss
        self.loss_fn = loss_fn
        self.loss_fn.to(self.device)

        # Set up optimizer(s)
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs

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
        Creates an InferenceAgent out of raw arguments
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
        # Set up loss
        loss_name = trainer_config.get("loss", "BCEWithLogitsLoss")
        loss_kwargs = trainer_config.get("loss_kwargs", {})
        loss_fn = Trainer.create_loss(loss_name, loss_kwargs)

        # Set up optim class but don't initialise
        optim_name = trainer_config.get("optimizer", "SGD")
        optim_kwargs = trainer_config.get("optimizer_kwargs", {"lr": 0.001})
        optimizer_class = Trainer.create_optimizer(optim_name)

        return super().create_inference_agent(
            data_path=data_path,
            data_map_path=data_map_path,
            out_dir=out_dir,
            exp_name=exp_name,
            trainer_config=trainer_config,
            model_config=model_config,
            classes=classes,
            checkpoint_path=checkpoint_path,

            loss_fn=loss_fn,
            optim_class=optimizer_class,
            optim_kwargs=optim_kwargs,
            shot_list=trainer_config.get("shot_list", (1, 4, 8, 12, 16, 20, 24)),
            reps_per_shot=trainer_config.get("reps_per_shot", 50),
            num_trials=trainer_config.get("num_trials", 10),
        )

    @staticmethod
    def create_dataset(classifier_name, data_path, data_map_path,
                       classes, interest_classes, transforms):
        """
        Creates a CropDataset
        @param data_path: Path to directory containing datasets
        @param data_map_path: Path to .json file containing dataset split information
        @param classes: JSON file containing class name --> class id
        @param interest_classes: List of specific classes (subset) to run experiment
        @param transforms: Dictionary of transform names to use for data augmentation
        @return: CropDataset
        """
        return TaskDataset(
            data_path=data_path,
            classes=classes,
            interest_classes=interest_classes,
            data_map_path=data_map_path,
            transforms=transforms,
            train_val_test=(0.8, 0.1, 0.1),
            use_one_hot=True,
            inf_mode=True
        )

    def format_and_compute_loss(self, preds, targets):
        """
        Wrapper function for formatting preds and targets for loss.
        @param preds: (b, #c, h, w) shape tensor of model outputs, #c is num classes
        @param targets: (b, 1 or #c, h, w) shape tensor of targets.
                        If only 1 channel, then squeeze it for CrossEntropy
        @return: loss value
        """
        return compute_masked_loss(self.loss_fn, preds, targets, invalid_value=-1)

    def evaluate_batch(self, preds, labels):
        """
        Returns confusion matrix for each class across batch
        @param preds: (b, #classes, h, w) Tensor of model predictions
        @param labels: (b, #classes, h, w) Tensor of labels
        @return: Confusion matrix metrics totalled across batch
        """
        # TODO: Fix inference for time series
        # Convert from tensor to numpy array
        # imgs = input_t.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        label_masks = labels.detach().cpu().numpy()

        batch_total_metrics = Counter()
        # TODO: Add back images to this loop later
        # Iterate over each image in batch.
        for ind, (pred, label_mask) in enumerate(zip(preds, label_masks)):
            _pred = pred[np.newaxis, ...]  # shape (b, #c, h, w)
            _label_mask = label_mask[np.newaxis, ...]  # shape (b, #c, h, w)

            # Get raw confusion matrix
            CM = confusion_matrix(_pred, _label_mask, pred_threshold=0)

            # Find the count of each class in ground truth,
            # record in metrics dict as whole num
            n = self.dataset.num_classes
            label_class_counts = np.count_nonzero(label_mask.reshape(n, -1), axis=-1)

            # Create metrics dict
            metrics_dict = create_metrics_dict(
                self.dataset.remapped_classes,
                tn=CM[:, 0, 0], fp=CM[:, 0, 1],
                fn=CM[:, 1, 0], tp=CM[:, 1, 1],
                gt_class_count=label_class_counts.tolist()
            )
            batch_total_metrics.update(metrics_dict)

        return batch_total_metrics

    def infer(self, set_type):
        """
        Runs inference for the model on the given set_type for the dataset.
        Saves metrics in inference out directory per ground truth mask.
        @param set_type: One of train/val/test
        """
        train_loaders, val_loaders, test_loaders = \
            self.dataset.create_data_loaders(batch_size=self.batch_size)
        loaders = {'train': train_loaders, 'val': val_loaders, 'test': test_loaders}
        data_loaders = loaders[set_type]

        # Begin meta inference
        metric_results_per_shot = {}
        for trial_num in range(1, self.num_trials + 1):
            # Shuffle tasks for each trial, but not across shots
            task_names = random.sample(data_loaders.keys(), len(data_loaders))

            for shots in self.shot_list:
                copy_model = deepcopy(self.model)

                # Set up optimizer for each run
                weights = filter(lambda w: w.requires_grad, copy_model.parameters())
                optim_kwargs = self.optim_kwargs if self.optim_kwargs is not None else {}
                optim_kwargs["lr"] = optim_kwargs.get("lr", 0.001) * np.sqrt(shots)
                optimizer = self.optim_class(weights, **optim_kwargs)

                # Accumulate the shots and keep track of how many shots consumed
                i = 0
                input_shots, labels = [], []
                while i < shots:
                    for task_name in task_names:
                        support_loader, _ = data_loaders[task_name]
                        for batch_index, (input_t, y) in enumerate(support_loader):
                            input_shots.append(input_t)  # shape (1, c, h, w)
                            labels.append(y)

                            i += 1
                            if i == shots:
                                break
                        else:
                            # Didn't break out of inner loop, still shots to go
                            continue

                        break  # Did break out of inner loop, so shots are done

                # Concatenate input_shots along batch dimension
                input_shots = tuple(zip(*input_shots))  # transposes the time and batch dimensions
                input_shots = [torch.cat(x_t, dim=0) for x_t in input_shots]
                labels = torch.cat(labels, dim=0)

                # Input into the model r times, r = num updates per shot
                copy_model.train()
                avg_loss = MeanMetric()
                for rep in range(self.reps_per_shot):
                    # Shift to correct device
                    input_shots, labels = \
                        self.dataset.shift_sample_to_device((input_shots, labels), self.device)

                    preds = copy_model(input_shots)

                    loss = self.format_and_compute_loss(preds, labels)
                    avg_loss.update(loss.item())

                    loss.backward()
                    optimizer.step()

                print(f"Shots batch loss after {i} shots: {avg_loss.item()}")

                # Now do inference on all of query data
                copy_model.eval()
                total_metrics = Counter()
                avg_loss = MeanMetric()
                for task_name, (support_loader, query_loader) in data_loaders.items():
                    for batch_index, (input_t, y) in enumerate(query_loader):
                        # Shift to correct device
                        input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

                        with torch.no_grad():
                            # Input into the model and get loss
                            preds = copy_model(input_t)

                            loss = self.format_and_compute_loss(preds, y)
                            avg_loss.update(loss.item())

                        # Update metrics across all tasks' query sets
                        batch_metrics_dict = self.evaluate_batch(preds, y)
                        total_metrics.update(batch_metrics_dict)

                if i not in metric_results_per_shot:
                    metric_results_per_shot[i] = [total_metrics]
                else:
                    metric_results_per_shot[i].append(total_metrics)

                print(f"Query loss after {i} shots: {avg_loss.item()}")

            # Save eval results by averaging
            # avg_results_per_shot = {
            #     i: {m_name: m_val/self.num_trials for m_name, m_val in metric_counts.items()}
            #     for i, metric_counts in metric_results_per_shot.items()
            # }

            # Save results every trial
            with open(os.path.join(self.out_dir, f"{self.exp_name}_shot_curve.json"), 'w') as f:
                json.dump(metric_results_per_shot, f, indent=2)
