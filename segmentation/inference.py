import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import logging
import argparse
import numpy as np
from PIL import Image, ImageDraw
from train import val_step
from segmentation import load_model, save_model
from data_loader import get_data_loaders, ConfigHandler
from metrics import calculate_iou_prec_recall, MeanMetric
from torch.utils.tensorboard import SummaryWriter


def draw_mask_on_im(img, masks):
  """
  Helper method that opens an image, draws the segmentation masks in `masks`
  as bitmaps, and then returns the masked image.
  Requires:
    `img`: np array of shape (#c, h, w)
    `masks`: Array shaped as: #C x h x w
  """
  # Open the image and set up an ImageDraw object
  im = Image.fromarray(img).convert('RGB')
  im_draw = ImageDraw.Draw(im)

  # Draw the bitmap for each class
  for i, mask in enumerate(masks):
    mask_im = Image.fromarray(mask.astype(np.uint8) * 64, mode='L')
    im_draw.bitmap((0, 0), mask_im, fill=(i, i, i))
  
  return im


def passed_arguments():
  parser = argparse.ArgumentParser(description=\
    "Script to run inference for segmentation models.")
  parser.add_argument("-d", "--data_path",
                      type=str,
                      required=True,
                      help="Path to directory containting datasets.")
  parser.add_argument("-c", "--config",
                      type=str,
                      required=True,
                      help="Path to .json model config file.")
  parser.add_argument("-s", "--set_type",
                      type=str,
                      default="val",
                      help="One of the train/val/test sets to perform inference.")
  parser.add_argument("--classes",
                      type=str,
                      default="classes.json",
                      help="Path to .json index->class name file.")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()

  set_type = args.set_type.lower()
  assert set_type in {"train", "val", "test"}, "Only train/val/test sets permitted."
  
  # Create config handler
  ch = ConfigHandler(args.data_path, args.config, args.classes)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # Load model
  model = load_model(ch, from_checkpoint=True)
  model.to(device)
  model.eval()

  # Create dataset loaders for inference.
  b_size = ch.config.get("batch_size", 32)
  train_loader, val_loader, test_loader = \
    get_data_loaders(ch, inf_mode=True, batch_size=b_size)
  
  loaders = {"train":train_loader, "val":val_loader, "test":test_loader}
  loader = loaders[args.set_type]

  # Set up metrics
  epoch_ious = MeanMetric()
  epoch_prec = MeanMetric()
  epoch_recall = MeanMetric()
  
  ## Begin inference
  for batch_index, (input_t, y) in enumerate(loader):
    # Shift to correct device
    input_t, y = input_t.to(device), y.to(device)

    # Input into the model
    preds = model(input_t)

    # Convert from tensor to numpy array
    imgs = input_t.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    label_masks = y.detach().cpu().numpy()

    # Iterate over each image in batch.
    for ind, (img, pred, label_mask) in enumerate(zip(imgs, preds, label_masks)):
      _pred = pred[np.newaxis, ...]
      _label_mask = label_mask[np.newaxis, ...]

      iou, prec, recall = calculate_iou_prec_recall(_pred, _label_mask, pred_threshold=0)

      # Create metrics dict
      metrics_dict = {'mean_iou': np.mean(iou),
                     'mean_prec': np.mean(prec),
                     'mean_recall': np.mean(recall)}
      
      # Break down IoU, precision and recall by class
      for i, class_name in ch.classes.items():
        sub_metric_dict = {'iou':iou, 'prec':prec, 'recall':recall}
        for metric_type, metrics in sub_metric_dict.items():
          class_metric_name = f'class_{class_name}/{metric_type}'
          class_metric = metrics[int(i)-1]  # i-1 to make indices 0 based
          metrics_dict[class_metric_name] = class_metric

      # Id for saving file.
      img_id = (batch_index * b_size) + ind

      # Convert from b,g,r to r,g,b
      img = img.astype(np.uint8)[1:4, ...]
      img = np.roll(img, 2, axis=0)
      img = img.transpose(1, 2, 0)

      # Save original image
      im = Image.fromarray(img).convert('RGB')
      im.save(os.path.join(ch.inf_dir, f"{img_id}_im.jpg"))

      # Draw pred mask on image
      pred_im = draw_mask_on_im(img, pred)
      pred_im.save(os.path.join(ch.inf_dir, f"{img_id}_pred.jpg"))

      # Draw ground truth mask on image
      gt_im = draw_mask_on_im(img, label_mask)
      gt_im.save(os.path.join(ch.inf_dir, f"{img_id}_gt.jpg"))

      # Save eval results.
      with open(os.path.join(ch.inf_dir, f"{img_id}_metrics.json"), 'w') as f:
        json.dump(metrics_dict, f)

  print(f"Inference complete for set {set_type}!")