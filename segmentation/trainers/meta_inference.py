import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import Counter

from utils.colors import get_color_choice
from data_loaders.dataset import CropDataset
from metrics import create_metrics_dict, confusion_matrix

from trainers.trainer import Trainer
from trainers.inference import InferenceAgent
from trainers.utils import create_dirs, load_model, create_dataset, compute_masked_loss


class MetaInferenceAgent:
    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            out_dir: str,
            loss_fn,
            optim_class,
            optim_kwargs,
            batch_size: int = 1,
            max_shots: int = 25,
            metric_names=(),
    ):
        """
        Handles inference, including saving metrics and TODO: images
        @param model: Segmentation model to evaluate
        @param dataset: CropDataset instance
        @param out_dir: Output directory where inference results will be saved
        @param batch_size: Batch size of input images for inference (default 1)
        @param max_shots: Max number of input shots before inference is performed
        @param metric_names: Names of metrics that will measure inference performance
        """
        self.batch_size = batch_size
        self.max_shots = max_shots
        self.out_dir = out_dir

        # Get list of metrics to use for training
        self.metric_names = metric_names
        if len(self.metric_names) == 0:
            self.metric_names = ["iou", "prec", "recall"]

        # Set up dataset
        self.dataset = dataset

        # Set up available device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Get model
        self.model = model
        self.model.to(self.device)

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
        # Create output directory, save directory and metrics directories.
        out_dir = os.path.join(
            out_dir if out_dir is not None else os.path.join(data_path, 'inference'),
            exp_name if exp_name is not None else
            "_".join((model_config["name"], trainer_config["name"]))
        )
        create_dirs(out_dir)

        # SAVE config file in output directory at beginning of inference
        InferenceAgent.save_config(trainer_config, out_dir, 'trainer_config')
        InferenceAgent.save_config(model_config, out_dir, 'model_config')

        # Set up loss
        loss_name = trainer_config.get("loss", "CrossEntropyLoss")
        loss_kwargs = trainer_config.get("loss_kwargs", {})
        loss_fn = Trainer.create_loss(loss_name, loss_kwargs)

        # Set up optim class but don't initialise
        optim_name = trainer_config.get("optimizer", "Adam")
        optim_kwargs = trainer_config.get("optimizer_kwargs", {"lr": 0.001})
        optimizer_class = Trainer.create_optimizer(optim_name)

        # Set up dataset
        classifier_name = model_config["classifier"].lower()
        interest_classes = trainer_config.get("interest_classes", [])
        transforms = trainer_config.get("transforms", {})
        dataset = cls.create_dataset(classifier_name, data_path, data_map_path,
                                     classes, interest_classes, transforms)

        # TODO: Find a way to break this link between model and trainer config
        # Set up model using its config file and number of classes from trainer config.
        num_classes = len(interest_classes or classes.keys())
        model = load_model(model_config, num_classes, checkpoint_path)

        return cls(
            model=model,
            dataset=dataset,
            batch_size=trainer_config.get("batch_size", 1),
            out_dir=out_dir,
            loss_fn=loss_fn,
            optim_class=optimizer_class,
            optim_kwargs=optim_kwargs,
            metric_names=trainer_config.get("metrics", []),
            **kwargs
        )

    @staticmethod
    def create_dataset(classifier_name, data_path, data_map_path,
                       classes, interest_classes, transforms):
        """
        Creates a CropDataset
        @param classifier_name: Name of model classifier, which determines which dataset to use
        @param data_path: Path to directory containing datasets
        @param data_map_path: Path to .json file containing dataset split information
        @param classes: JSON file containing class name --> class id
        @param interest_classes: List of specific classes (subset) to run experiment
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
        for shots in range(1, self.max_shots+1):
            copy_model = deepcopy(self.model)

            # Set up optimizer for each run
            weights = filter(lambda w: w.requires_grad, copy_model.parameters())
            optim_kwargs = {} if self.optim_kwargs is None else self.optim_kwargs
            optimizer = self.optim_class(weights, **optim_kwargs)

            # Randomly pick a task
            task_names = random.sample(data_loaders.keys(), len(data_loaders))

            # Keep track of how many shots consumed
            i = 0
            total_loss = 0.
            copy_model.train()
            while i < shots:
                for task_name in task_names:
                    support_loader, _ = data_loaders[task_name]
                    for batch_index, (input_t, y) in enumerate(support_loader):
                        # Shift to correct device
                        input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

                        # Input into the model
                        preds = self.model(input_t)

                        loss = self.format_and_compute_loss(preds, y)
                        total_loss += loss

                        i += 1
                        if i == shots:
                            break
                    else:
                        # Didn't break out of inner loop, still shots to go
                        continue

                    break  # Did break out of inner loop, so shots are done

            loss = total_loss / i
            print(f"Loss after {i} shots: {loss.item()}")
            loss.backward()
            optimizer.step()

            # Now do inference on all of query data
            copy_model.eval()
            total_metrics = Counter()
            for task_name, (support_loader, query_loader) in data_loaders.items():
                for batch_index, (input_t, y) in enumerate(query_loader):
                    # Shift to correct device
                    input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

                    # Input into the model
                    preds = self.model(input_t)

                    # TODO: Fix inference for time series
                    # Convert from tensor to numpy array
                    # imgs = input_t.detach().cpu().numpy()
                    preds = preds.detach().cpu().numpy()
                    label_masks = y.detach().cpu().numpy()

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

                        # Update metrics across all tasks' query sets
                        total_metrics.update(metrics_dict)

            metric_results_per_shot[i] = total_metrics

            # Save eval results
            with open(os.path.join(self.out_dir, f"meta_inference_shot_curve.json"), 'w') as f:
                json.dump(metric_results_per_shot, f, indent=2)
