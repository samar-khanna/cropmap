import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_color_choice(i):
  """
  Creates an `(r, g, b)` tuple (values from 0 to 255) for a given index. \n
  Requires: \n
    `i`: Any non-negative integer. \n
  Returns: \n
    `color`: An `(r, g, b)` tuple of 8-bit values from 0 to 255.
  """
  sh = lambda m: (i << m) % 255 
  color_choice = {
    0: (255, sh(6), sh(3)), 1: (sh(6), 255, sh(3)), 2:(sh(6), sh(3), 255),
    3: (255, sh(2), sh(4)), 4: (sh(2), 255, sh(4)), 5: (sh(2), sh(4), 255),
    6: (255, 255, sh(3)), 7:(255, sh(3), 255), 8:(sh(3), 255, 255)
  }
  return color_choice.get(i % 9)


def plot_color_legend(classes, save_image=False):
  """
  Plots color legend for each class in `classes`.
  Requires:
    `classes`: Dictionary of `class_name` --> `class_index` \n
  """
  handles = []
  for class_name, class_idx in classes.items():
    r, g, b = get_color_choice(class_idx)
    color = r/255, g/255, b/255
    patch = mpatches.Patch(color=color, label=class_name)

    handles.append(patch)
  
  plt.legend(
    handles=handles, 
    ncol=5, 
    mode="expand", 
    fontsize="x-small",
    title="Color legend")
  plt.show()