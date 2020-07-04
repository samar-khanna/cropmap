import numpy as np


class MeanMetric():
  """
  Handler to keep track of running average metrics.
  Data stored as numpy arrays.
  """
  def __init__(self, data=None):
    super().__init__()
    self.n = 0
    self.data = data

  def update(self, vals):
    """
    Updates the `data` field with the input values.
    Requires:
      vals is an np array.
    """
    # Update average
    if self.data is not None:
      self.data = (self.data * self.n) + vals
      self.data = self.data/(self.n+1)
    else:
      self.data = vals
    self.n += 1
  
  def reset(self):
    """
    Resets `data` and `n`.
    """
    self.n = 0
    self.data = None
  
  def item(self):
    """
    Returns value in `data`, if present.
    """
    if self.data is None:
      raise ValueError("No data in this metric yet. Add data with .update()")
    return self.data


def calculate_metrics(preds, label_masks, pred_threshold=0.0):
  """
  Calculate IoU, Precision and Recall per class for entire batch of images.
  Requires:
    preds: model preds array, shape        (batch, #c, h, w)
    label_masks: ground truth masks, shape (batch, #c, h, w)
    pred_threshold: Confidence threshold over which pixel prediction counted.
  Returns:
    ious, precs, recall, kappa per class: shape (#c)
  """
  # Change view so that shape is (batch, h, w, #c)
  preds = preds.transpose(0, 2, 3, 1)
  label_masks = label_masks.transpose(0, 2, 3, 1)

  # Reduce dimensions across all but classes dimension.
  preds = preds.reshape(-1, preds.shape[-1])
  label_masks = label_masks.reshape(-1, label_masks.shape[-1])

  preds = preds > pred_threshold
  intersection = np.logical_and(preds, label_masks)
  union = np.logical_or(preds, label_masks)
  iou_scores = np.sum(intersection, axis=0) / np.sum(union, axis=0)
  iou_scores[np.isnan(iou_scores)] = 0.0

  precision = np.sum(intersection, axis=0)/np.sum(preds, axis=0)
  precision[np.isnan(precision)] = 0.0

  recall = np.sum(intersection, axis=0)/np.sum(label_masks, axis=0)
  recall[np.isnan(recall)] = 0.0

  metrics = {"iou": iou_scores, "prec": precision, "recall": recall}
  return metrics