import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import logging
import argparse
import numpy as np
from segmentation import load_model, save_model
from data_loader import get_data_loaders, ConfigHandler
from metrics import calculate_iou_prec_recall, MeanMetric
from torch.utils.tensorboard import SummaryWriter


def get_loss_optimizer(config, model):
  """
  Instantiates loss function and optimizer based on name and kwargs.
  Ensure that names are valid in the torch.nn and torch.optim modules.
  Also ensure keyword arguments match.
  Defaults to using BinaryCrossentropy (from logits), and Adam(lr=0.0001)
  """ 
  # Set up loss.
  loss_name = config.get("loss", "BCEWithLogitsLoss")
  loss_kwargs = config.get("loss_kwargs", {})
  assert loss_name in nn.__dict__,\
    "Invalid PyTorch loss. The name must exactly match a loss in the nn module"
  loss_fn = nn.__dict__[loss_name](**loss_kwargs)

  # Set up optimizer
  optim_name = config.get("optimizer", "Adam")
  optim_kwargs = config.get("optimizer_kwargs", {"lr":0.001})
  assert optim_name in optim.__dict__,\
    "Invalid PyTorch optimizer. The name must exactly match an optimizer in the optim module"
  optimizer = optim.__dict__[optim_name](model.parameters(), **optim_kwargs)
  return loss_fn, optimizer


def log_metrics(metrics_dict, writer, epoch, phase):
  """
  Logs metrics to tensorflow summary writers, and to a log file.
  Also prints mean metrics for the epoch
  Requires:
    metrics_dict: Pairs of metric_name (make it informative!), metric_value_number
    writer: Either training or validation summary_writer
    phase: Either 'train' or 'val'
  """
  # Do it separately for tf writers.
  for metric_name, metric_value in metrics_dict.items():
    writer.add_scalar(metric_name, metric_value, global_step=epoch+1)

  print(f"Phase: {phase}")
  for metric_name, metric_value in metrics_dict.items():
    logging.info(f"Epoch {epoch+1}, Phase {phase}, {metric_name}: {metric_value}")
    
    if not metric_name.startswith('class'):
      print(f"{metric_name}: {metric_value}")


def train_step(model, loss_fn, optimizer, images, labels):
  """
  Performs one training step over a batch.
  Passes the batch of images through the model, and backprops the gradients.
  Returns the resulting model predictions and loss values.
  """
  # Flush the gradient buffers
  optimizer.zero_grad()
  
  # Feed model
  preds = model(images)
  loss = loss_fn(preds, labels)

  # Backpropagate
  loss.backward()
  optimizer.step()

  return preds, loss

def val_step(model, loss_fn, optimizer, images, labels):
  """
  Performs one validation step over a batch.
  Passes the batch of images through the model.
  Returns the resulting model predictions and loss values.
  """
  # Feed model
  preds = model(images)
  loss = loss_fn(preds, labels)

  return preds, loss


def passed_arguments():
  parser = argparse.ArgumentParser(description=\
    "Script to train segmentation models.")
  parser.add_argument("-d", "--data_path",
                      type=str,
                      required=True,
                      help="Path to directory containting datasets.")
  parser.add_argument("-c", "--config",
                      type=str,
                      required=True,
                      help="Path to .json model config file.")
  parser.add_argument("--classes",
                      type=str,
                      default="classes.json",
                      help="Path to .json index->class name file.")
  parser.add_argument("-ckpt", "--from_checkpoint",
                      action="store_true",
                      default=False,
                      help="Whether to load model weights from checkpoint file.")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()
  
  # Create config handler
  ch = ConfigHandler(args.data_path, args.config, args.classes)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  ## TODO: Finetuning (freezing layers)
  # Load model
  model = load_model(ch, from_checkpoint=args.from_checkpoint)
  model.to(device)

  # Load optimizer and loss
  loss_fn, optimizer = get_loss_optimizer(ch.config, model)

  ## Set up Data Loaders.
  epochs = ch.epochs
  b_size = ch.config.get("batch_size", 32)
  train_loader, val_loader, _ = get_data_loaders(ch, inf_mode=False, batch_size=b_size)

  ## Set up tensorboards
  metrics_path = ch.metrics_dir
  train_writer = SummaryWriter(log_dir=os.path.join(metrics_path, 'train'))
  val_writer = SummaryWriter(log_dir=os.path.join(metrics_path, 'val'))
  logging.basicConfig(filename=os.path.join(metrics_path, "train_log.log"), level=logging.INFO)

  ## Begin training
  best_val_iou = -np.inf
  for epoch in range(epochs):
    for phase in ["train", "val"]:
      if phase == "train":
        writer = train_writer
        loader = train_loader
        feed_model = train_step
        model.train()
      else:
        writer = val_writer
        loader = val_loader
        feed_model = val_step
        model.eval()

      # Set up metrics
      epoch_loss = MeanMetric()
      epoch_ious = MeanMetric()
      epoch_prec = MeanMetric()
      epoch_recall = MeanMetric()

      for batch_index, (input_t, y) in enumerate(loader):
        # Shift to correct device
        input_t, y = input_t.to(device), y.to(device)

        # Input into the model
        preds, loss = feed_model(model, loss_fn, optimizer, input_t, y)

        # Conver to numpy and calculate metrics
        preds_arr = preds.detach().cpu().numpy()
        y_arr = y.detach().cpu().numpy()
        ious, prec, recall = calculate_iou_prec_recall(preds_arr, y_arr, pred_threshold=0)

        # Update metrics
        epoch_loss.update(loss.item())
        epoch_ious.update(ious)
        epoch_prec.update(prec)
        epoch_recall.update(recall)
      
      # Create metrics dict
      metrics_dict = {'epoch_loss': epoch_loss.item(),
                     'mean_iou': np.mean(epoch_ious.item()),
                     'mean_prec': np.mean(epoch_prec.item()),
                     'mean_recall': np.mean(epoch_recall.item())}
      
      # Break down IoU, precision and recall by class
      for i, class_name in ch.classes.items():
        sub_metric_dict = {'iou':epoch_ious, 'prec':epoch_prec, 'recall':epoch_recall}
        for metric_type, metrics in sub_metric_dict.items():
          metrics = metrics.item()
          class_metric_name = f'class_{class_name}/{metric_type}'
          class_metric = metrics[int(i)-1]  # i-1 to make indices 0 based
          metrics_dict[class_metric_name] = class_metric

      # Log metrics
      log_metrics(metrics_dict, writer, epoch, phase)

      # Save model checkpoint if val iou better than best recorded so far.
      if phase == "val":
        if metrics_dict['mean_iou'] > best_val_iou:
          save_model(model, ch)
      
