import os
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


def create_metrics_dict(classes, loss=None, **metrics):
  """
  Creates a metrics dictionary, mapping `metric_name`--> `metric_val`. \n
  Requires: \n
    `classes`: Dictionary of `class_name` --> index in metric array \n
    `loss`: Either `None`, or if specified, PyTorch loss number for the epoch \n
    `metrics`: Each metric should be a numpy array of shape (num_classes) \n
  """
  metrics_dict = {}

  if loss:
    metrics_dict["epoch_loss"] = loss

  # First log mean metrics
  for metric_name, metric_arr in metrics.items():
    metric = metric_arr
    metrics_dict[f"mean/{metric_name}"] = np.mean(metric)

    # Break down metric by class
    for class_name, i in classes.items():
      class_metric_name = f'class_{class_name}/{metric_name}'
      class_metric = metric[i]
      metrics_dict[class_metric_name] = class_metric
  
  return metrics_dict


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
                      default=os.path.join("segmentation", "classes.json"),
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
  device = torch.device("cuda:0" if use_cuda else "cpu")

  ## TODO: Finetuning (freezing layers)
  # Load model, use DataParallel if more than 1 GPU available
  model = load_model(ch, from_checkpoint=args.from_checkpoint)
  if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
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
  logging.basicConfig(filename=os.path.join(metrics_path, "log.log"), level=logging.INFO)

  ## Begin training
  best_val_iou = -np.inf
  for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}:")
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
      metrics_dict = create_metrics_dict(
        ch.classes,
        loss=epoch_loss.item(),
        iou=epoch_ious.item(),
        prec=epoch_prec.item(),
        recall=epoch_recall.item()
      )

      # Log metrics
      log_metrics(metrics_dict, writer, epoch, phase)

      # Save model checkpoint if val iou better than best recorded so far.
      if phase == "val":
        if metrics_dict['mean_iou'] > best_val_iou:
          print(f"Saving weights...")
          if torch.cuda.device_count() > 1:
            save_model(model.module, ch)
          else:
            save_model(model, ch)
    
    print(f"Finished epoch {epoch+1}.\n")
      