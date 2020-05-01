import os
import json
import argparse
import numpy as np
from metrics import MeanMetric
import matplotlib.pyplot as plt


def passed_arguments():
  parser = argparse.ArgumentParser(description=\
    "Script to run inference for segmentation models.")
  parser.add_argument("-inf", "--inference",
                      type=str,
                      required=True,
                      help="Path to directory containing inference results.")
  parser.add_argument("--classes",
                      type=str,
                      default="classes.json",
                      help="Path to .json index->class name file.")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()

  inf_path = args.inference
  metric_files = [os.path.join(inf_path, f) for f in os.listdir(inf_path) 
                  if f.endswith('.json')]
  print(f"Number of images evaluated: {len(metric_files)}")

  # Calculate mean metrics across all images in inference out.
  mean_results = {}
  for i, metric_path in enumerate(metric_files):
    with open(metric_path, 'r') as f:
      metrics = json.load(f)
    
    for metric_name, val in metrics.items():
      if metric_name not in mean_results:
        mean_results[metric_name] = MeanMetric(val)
      else:
        mean_results[metric_name].update(val)
  
  print("Evaluation results:")
  for metric_name, metric_val in mean_results.items():
    print(f"{metric_name}: {round(metric_val.item(), 3)}")

  pass