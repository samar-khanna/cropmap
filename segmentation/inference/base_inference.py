import os
import json
import numpy as np
import torch
import torch.nn as nn

from data_loaders.dataset import CropDataset

from utils.loading import load_model, create_dataset


class InferenceAgent:

    _SAVED_MODEL_CONFIG_NAME = "model_config"
    _SAVED_TRAINER_CONFIG_NAME = "inference_config"
    _DEFAULT_BATCH_SIZE = 32

    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            batch_size: int,
            out_dir: str,
            exp_name: str,
            metric_names=(),
    ):
        """
        Base abstract class for running inference.
        @param model: Segmentation model to evaluate
        @param dataset: CropDataset instance
        @param batch_size: Batch size of input images for inference
        @param out_dir: Output directory where inference results will be saved
        @param exp_name: Unique name for inference experiment
        @param metric_names: Names of metrics that will measure inference performance
        """
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.exp_name = exp_name  # TODO: We don't need this. Get rid.

        # Get list of metrics to use for training
        self.metric_names = metric_names
        if len(self.metric_names) == 0:
            self.metric_names = ["iou", "prec", "recall", "class_acc"]

        # Set up dataset
        self.dataset = dataset

        # Set up available device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Get model
        self.model = model
        self.model.to(self.device)
        self.model.eval()

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
        @param exp_name: Unique name for inference experiment
        @param trainer_config: JSON config file containing InferenceAgent parameters
        @param model_config: JSON config file containing Model parameters
        @param classes: JSON file containing Cropmap class name --> class id
        @param checkpoint_path: Path to .bin checkpoint file containing model weights
        @param kwargs: Extra keyword arguments to pass to init function.
        @return: Initialised Inference Agent
        """
        # Create output directory, save directory and metrics directories.
        out_dir = out_dir if out_dir is not None else \
            os.path.join(data_path, 'inference', "_".join((model_config["name"], trainer_config["name"])))
        os.makedirs(out_dir, exist_ok=True)

        # SAVE config file in output directory at beginning of inference
        cls.save_config(trainer_config, out_dir, cls._SAVED_TRAINER_CONFIG_NAME)
        cls.save_config(model_config, out_dir, cls._SAVED_MODEL_CONFIG_NAME)

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
            batch_size=trainer_config.get("batch_size", cls._DEFAULT_BATCH_SIZE),
            out_dir=out_dir,
            exp_name=exp_name,
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

    def infer(self, set_type, save_images=False):
        """
        Runs inference for the model on the given set_type for the dataset.
        @param set_type: One of train/val/test
        @param save_images: Whether to store pred/gt image results in out dir
        """
        raise NotImplementedError("This should be overwritten")
