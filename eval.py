import os
import re
import json
import random
import argparse
import numpy as np
from PIL import Image
from segmentation.metrics import MeanMetric
import matplotlib.pyplot as plt
from collections import OrderedDict


def passed_arguments():
  parser = argparse.ArgumentParser(description=\
    "Script to run inference for segmentation models.")
  parser.add_argument("-inf", "--inference",
                      type=str,
                      required=True,
                      help="Path to directory containing inference results.")
  parser.add_argument("--text",
                      action="store_true",
                      default=False,
                      help="Only print the text results from evaluation.")
  parser.add_argument("--classes",
                      type=str,
                      default=os.path.join("segmentation", "classes.json"),
                      help="Path to .json class->index name file.")
  args = parser.parse_args()
  return args


def sort_key(file_name):
  d = re.search('[0-9]+', file_name)
  return int(file_name[d.start():d.end()]) if d else float('inf')


def plot_pie(dist, title=None, thresh=0.02):
  """
  Creates and plots a pie chart, intended to represent distribution of
  classes in a dataset, for example.
  Requires:
    dist: a dictionary of name --> count
    thresh: percentage threshold above which metric will not be lumped into "other"
  """
  total = sum(dist.values())
  dist = {n: v/total for n, v in dist.items()}

  _dist = {"other":0}
  for n, val in dist.items():
    if val > thresh:
      _dist[n] = val
    else:
      _dist["other"] += val
  
  values = _dist.values()
  explode = [0.1*v for v in values]
  labels = _dist.keys()

  fig, ax = plt.subplots()
  ax.pie(values, explode=explode, labels=labels, 
         autopct="%1.1f%%", labeldistance=1.05)
  ax.axis("equal") # Equal aspect ratio ensures that pie is drawn as a circle.

  if title:
    ax.set_title(title, pad=10)
  fig.show()


def plot_hist(metrics, thresh=0.01):
  """
  Plots a histogram on the mean results of the inference task.
  """
  classes_metrics = {}
  seen = {}
  for metric_name, metric_val in metrics.items():
    class_name = metric_name.split('/')[0]
    metric_type = metric_name.split('/')[-1]
    class_name = class_name.replace("class_", "")

    # Store metric results only for mean metrics and metrics above threshold
    class_results = classes_metrics.get(class_name, OrderedDict())
    if class_results or class_name.find("mean") > -1 or metric_val > thresh:
      if class_name in seen:
        seen_results = seen[class_name]
        class_results.update(seen_results)
        del seen[class_name]

      class_results[metric_type] = metric_val
      classes_metrics[class_name] = class_results
    else:
      seen_results = seen.get(class_name, OrderedDict())
      seen_results[metric_type] = metric_val
      seen[class_name] = seen_results

  # Keep mean results first.
  sort_keys = {"mean": 0, "corn": 1, "soybeans": 2}
  sort_key = lambda k: sort_keys.get(k.lower(), len(sort_keys))
  x_labels = sorted(classes_metrics.keys(), key=sort_key)
  x = np.arange(len(x_labels))
  
  # List of rectangles for each metric type
  metric_type_results = {}
  for i, label in enumerate(x_labels):
    class_results = classes_metrics[label]
    
    # Unrol metric values into lists for each metric type 
    for metric_type, metric_val in class_results.items():
      rects = metric_type_results.get(metric_type, [])
      rects.append(metric_val)

      metric_type_results[metric_type] = rects
  
  n = len(metric_type_results)  # Number of metrics per class

  ## Set up plot
  fig, ax = plt.subplots()
  width = 0.6  # Combined bar width for each label
  metric_type_rects = []
  for i, (metric_type, results) in enumerate(metric_type_results.items()):
    rect_pos = x - (width/2) + (width/(2 * n)) + (i * (width/n))
    rects = ax.bar(rect_pos, results, width/n, label=metric_type)
    metric_type_rects.append(rects)
  
  print(metric_type_rects)
  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate(f'{round(height, 3)}',
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')

  # Annotate the bars
  for rects in metric_type_rects:
    autolabel(rects)

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Metric Scores')
  ax.set_title('Metric scores by class')
  ax.set_xticks(x)
  ax.set_xticklabels(x_labels)
  ax.legend()

  fig.tight_layout()
  fig.show()
  plt.show()


def plot_images(im_paths):
  im_path, gt_path, gt_raw_path, pred_path, pred_raw_path, metric_path = im_paths

  with open(metric_path, 'r') as f:
    metrics = json.load(f)

  fig, axes = plt.subplots(nrows=2, ncols=3)
  fig.delaxes(axes[1,2])

  corn_iou = round(metrics['class_Corn/iou'], 3)
  soybean_iou = round(metrics['class_Soybeans/iou'], 3)
  fig.suptitle(f"Corn IoU: {corn_iou}, Soybean IoU: {soybean_iou}")

  im = Image.open(im_path).convert('RGB')
  gt = Image.open(gt_path).convert('RGB')
  pred = Image.open(pred_path).convert('RGB')
  gt_raw = Image.open(gt_raw_path).convert('RGB')
  pred_raw = Image.open(pred_raw_path).convert('RGB')

  axes[0, 0].imshow(im)
  axes[0, 1].imshow(pred)
  axes[0, 2].imshow(gt)
  axes[1, 0].imshow(pred_raw)
  axes[1, 1].imshow(gt_raw)

  plt.show()


if __name__ == "__main__":
  args = passed_arguments()

  inf_path = args.inference

  im_paths = []
  gt_paths = []
  gt_raw_paths = [] 
  pred_paths = [] 
  pred_raw_paths = [] 
  metric_paths = []

  sorted_files = sorted(os.listdir(inf_path), key=sort_key)
  for f in sorted_files:
    if f.find('im') > -1 and f.find('.jpg') > -1:
      im_paths.append(os.path.join(inf_path, f))
    elif f.find('raw_gt') > -1 and f.find('.jpg') > -1:
      gt_raw_paths.append(os.path.join(inf_path, f))
    elif f.find('gt') > -1 and f.find('.jpg') > -1:
      gt_paths.append(os.path.join(inf_path, f))
    elif f.find('raw_pred') > -1 and f.find('.jpg') > -1:
      pred_raw_paths.append(os.path.join(inf_path, f))
    elif f.find('pred') > -1 and f.find('.jpg') > -1:
      pred_paths.append(os.path.join(inf_path, f))
    elif f.find('metric') > -1 and f.find('.json') > -1:
      metric_paths.append(os.path.join(inf_path, f))
  print(f"Number of images evaluated: {len(im_paths)}")

  # Calculate mean metrics across all images in inference out.
  mean_results = {}
  for i, metric_path in enumerate(metric_paths):
    with open(metric_path, 'r') as f:
      metrics = json.load(f)
    
    for metric_name, val in metrics.items():
      if metric_name not in mean_results:
        mean_results[metric_name] = MeanMetric(val)
      else:
        mean_results[metric_name].update(val)
  
  # Get rid of MeanMetrics
  mean_results = {n: metric.item() for n, metric in mean_results.items()}

  print("Evaluation results for non-zero metrics:")
  for metric_name, metric_val in mean_results.items():
    if metric_val > 0:
      print(f"{metric_name}: {round(metric_val, 3)}")

  ## Start plotting results.
  if not args.text:
    # Mean result histogram
    plot_hist(mean_results, thresh=0.2)
    
    # Plot grids of the images.
    paths = zip(im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths)
    paths = list(paths)

    show_paths = random.sample(paths, 20)
    for ps in show_paths:
      plot_images(ps)
