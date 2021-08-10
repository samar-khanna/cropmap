import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from trainers.base_trainer import Trainer
from data_loaders.dataset import CropDataset
from trainers.trainer_utils import compute_masked_loss
from metrics import calculate_metrics, MeanMetric


class KNNTrainer(Trainer):
    def __init__(self):
        super().__init__(
            model,
            dataset,
            loss_fn,
            optim_class,
            num_shots,
            batch_size=batch_size,
            num_epochs=num_epochs,
            use_one_hot=use_one_hot,
            save_path=save_path,
            num_display=num_display,
            metric_names=metric_names,
            optim_kwargs=optim_kwargs,
            train_writer=train_writer,
            val_writer=val_writer,
        )
