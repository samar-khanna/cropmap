import numpy as np


class MeanMetric():
    """
    Handler to keep track of running average metrics.
    Data stored as numpy arrays, or numbers.
    """
    def __init__(self, data=None):
        super().__init__()
        self.n = 0
        self.data = None

        if data is not None:
            self.update(data)

    def update(self, vals):
        """
        Updates the `data` field with the input values.
        Requires:
          vals is an np array.
        """
        if np.isnan(vals).any():
            print("Warning: vals contains one or more NaN values.")
            return

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


def confusion_matrix(preds, ground_truth, pred_threshold=0.0):
    """
    Calculate Confusion Matrix per class for entire batch of images.
    Requires:
      preds: model preds array, shape        (batch, #c, h, w)
      ground_truth: ground truth masks, shape (batch, #c, h, w)
      pred_threshold: Confidence threshold over which pixel prediction counted.
    Returns:
      confusion matrix per class: shape (#c, 2, 2)
      (0,0): TN, (0,1): FP, (1,0): FN, (1,1): TP
    """
    # Change view so that shape is (batch, h, w, #c)
    preds = preds.transpose(0, 2, 3, 1)
    ground_truth = ground_truth.transpose(0, 2, 3, 1)

    # Reduce dimensions across all but classes dimension.
    preds = preds.reshape(-1, preds.shape[-1])
    ground_truth = ground_truth.reshape(-1, ground_truth.shape[-1])

    # Pred values are true if above threshold
    preds = preds > pred_threshold

    # Intersection is the true positives
    intersection = np.logical_and(preds, ground_truth)

    # Calculate confusion matrix for each class
    tp = np.sum(intersection, axis=0)
    fp = np.sum(preds, axis=0) - tp
    fn = np.sum(ground_truth, axis=0) - tp
    tn = ground_truth.shape[0] - tp - fp - fn

    # Confusion matrix shape is (#c, 2, 2)
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def calculate_metrics(preds, label_masks, pred_threshold=0.0, zero_nans=True):
    """
    Calculate IoU, Precision and Recall per class for entire batch of images.
    Requires:
      preds: model preds array, shape        (batch, #c, h, w)
      label_masks: ground truth masks, shape (batch, #c, h, w)
      pred_threshold: Confidence threshold over which pixel prediction counted.
      zero_nans: whether to zero out nan metrics.
    Returns:
      ious, precs, recall, kappa per class: shape (#c)
    """
    CM = confusion_matrix(preds, label_masks, pred_threshold=pred_threshold)
    tn, fp, fn, tp = CM[:, 0, 0], CM[:, 0, 1], CM[:, 1, 0], CM[:, 1, 1]

    # Dimensions of each of following arrays is (#c)
    iou_scores = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    if zero_nans:
        iou_scores[np.isnan(iou_scores)] = 0.0
        precision[np.isnan(precision)] = 0.0
        recall[np.isnan(recall)] = 0.0
        accuracy[np.isnan(accuracy)] = 0.0

    metrics = {"iou": iou_scores, "prec": precision, "recall": recall, "acc": accuracy}
    return metrics


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
        metrics_dict[f"mean/{metric_name}"] = np.nanmean(metric)

        # Break down metric by class
        for class_name, i in classes.items():
            class_metric_name = f'class_{class_name}/{metric_name}'
            class_metric = metric[i]
            metrics_dict[class_metric_name] = class_metric

    return metrics_dict
