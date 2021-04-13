import os
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn

from trainers.base_trainer import Trainer
from trainers.default_trainer import DefaultTrainer
from trainers.maml_trainer import MAMLTrainer
from trainers.simclr_trainer import SimCLRTrainer

TRAINER_TYPES = {
    "default": DefaultTrainer,
    "maml": MAMLTrainer,
    "simclr":SimCLRTrainer
}


def passed_arguments():
    parser = argparse.ArgumentParser(description="Script to train segmentation models.")
    parser.add_argument("-d", "--data_path",
                        type=str,
                        required=True,
                        help="Path to directory containing datasets.")
    parser.add_argument("-c", "--model",
                        type=str,
                        required=True,
                        help="Path to .json model config file.")
    parser.add_argument("--trainer",
                        type=str,
                        required=True,
                        help="Path to .json trainer config file.")
    parser.add_argument("--data_map",
                        type=str,
                        default=None,
                        help="Path to .json file with train/val/test split for experiment.")
    parser.add_argument("--out_dir",
                        type=str,
                        default=None,
                        help="Path to directory where model outputs will be stored.")
    parser.add_argument("--split",
                        nargs="+",
                        type=float,
                        default=[0.8, 0.1, 0.1],
                        help="Train/val/test split percentages.")
    parser.add_argument("--classes",
                        type=str,
                        default=os.path.join("segmentation", "classes.json"),
                        help="Path to .json index->class name file.")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=False,
                        help="Path to load model weights from checkpoint file.")
    parser.add_argument("--freeze_backbone",
                        action="store_true",
                        help="Whether to freeze backbone layers while training.")
    parser.add_argument("--new_head",
                        action="store_true",
                        help="Whether to allow non-strict loading with a new head")
    parser.add_argument("--start_epoch",
                        type=int,
                        default=0,
                        help="Start logging metrics from this epoch number.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = passed_arguments()

    # Classes is dict of {class_name --> class_id}
    with open(args.classes, 'r') as f:
        classes = json.load(f)

    # Contains config parameters for model
    with open(args.model, 'r') as f:
        model_config = json.load(f)

    # Contains config parameters for trainer
    with open(args.trainer, 'r') as f:
        trainer_config = json.load(f)

    trainer_type = trainer_config.get("trainer", "default").lower()
    assert trainer_type in TRAINER_TYPES, f"Unsupported trainer type {trainer_type}"

    # Instantiate trainer
    trainer = TRAINER_TYPES[trainer_type].create_trainer(
        args.data_path,
        args.out_dir,
        args.data_map,
        trainer_config,
        model_config,
        classes,
        args.checkpoint,
        args.freeze_backbone,
        args.new_head,
    )

    # Run training
    trainer.train(args.start_epoch)
