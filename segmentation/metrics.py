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


def reshape_channel_last(image):
    """
    Reshapes a batched image array to pixel-level vectors of the channel dimension
    @param image: A PyTorch style image np array, shape (b, #c, h, w)
    @return: A matrix shape (b*h*w, #c) representing channel info per pixel
    """
    image = image.transpose(0, 2, 3, 1)  # (b, #c, h, w) -> (b, h, w, #c)
    pixel_vecs = image.reshape(-1, image.shape[-1])  # (b, h, w, #c) -> (b*h*w, #c)
    return pixel_vecs


def mean_accuracy(preds, ground_truth, unk_pred_mask=None, unk_gt_mask=None,
                  unk_combined_mask=None):
    """
    Calculates mean class accuracy across all instances
    N := number of instances, #c := number of classes
    @param preds: Binary-valued prediction array. (N, #c)
    @param ground_truth: Binary-valued ground truth array. (N, #c)
    @param unk_pred_mask: Bool mask to indicate which preds are unk class. (N,)
    @param unk_gt_mask: Bool mask to indicate which ground truth are unk class. (N,)
    @param unk_combined_mask: Bool mask to indicate unk class instances in both pred/gt.
                              Will ignore unk_pred_mask and unk_gt_mask if provided.
    @return: Mean class accuracy (num_correct_instances/total_instances)
    """
    # Take argmax to get class predictions and ground truth indexes
    pixel_preds = np.argmax(preds, axis=1)  # (N, c) -> (N,)
    pixel_ground_truth = np.argmax(ground_truth, axis=1)  # (N, c) -> (N)

    if unk_combined_mask is not None:
        # Use combined mask to calculate accuracy only on known classes
        known_mask = np.logical_not(unk_combined_mask)
        pixel_preds = pixel_preds[known_mask]  # (n,)
        pixel_ground_truth = pixel_ground_truth[known_mask]  # (n,)
    else:
        # Unknown value is -1
        if unk_pred_mask is not None:
            pixel_preds[unk_pred_mask] = -1
        if unk_gt_mask is not None:
            pixel_ground_truth[unk_gt_mask] = -1

    if len(pixel_ground_truth) == 0:
        print("Warning: no known classes in batch")
        return 0.

    correct_mask = pixel_preds == pixel_ground_truth
    accuracy = np.sum(correct_mask)/len(correct_mask)
    return accuracy


def mean_accuracy_from_images(preds, ground_truth, pred_threshold=0.0):
    """
    Calculates mean class accuracy across a batch of predicted images.
    @param preds: model preds logits array,    (b, #c, h, w)
    @param ground_truth: one-hot ground truth, (b, #c, h, w)
    @param pred_threshold: Confidence threshold over which pixel prediction counted.
    @return: Mean class accuracy (num_correct_pixels/total_pixels)
    """
    # Reduce all dims except classes dim (b,c,h,w) -> (b,h,w,c) -> (b*h*w, c)
    preds = reshape_channel_last(preds)
    ground_truth = reshape_channel_last(ground_truth)

    # If pixel logit value across all classes is < 0 in pred, class is unknown
    # If pixel value across all classes is 0 in ground truth, class is unknown
    unk_pred_mask = np.all(preds <= pred_threshold, axis=1)  # (b*h*w,c) -> (b*h*w)
    unk_gt_mask = np.all(~ground_truth.astype(np.bool), axis=1)  # (b*h*w,c) -> (b*h*w)

    # TODO: Use of a combined mask to calculate accuracy on known classes? Or 2 masks?
    return mean_accuracy(preds, ground_truth, unk_combined_mask=unk_gt_mask)


def confusion_matrix(preds, ground_truth):
    """
    Calculates confusion matrix per class.
    Here total classes is #c, total number of instances is N
    @param preds: Binary-valued prediction array. (N, #c)
    @param ground_truth: Binary-valued ground truth array. (N, #c)
    @return: confusion matrix per class: shape (#c, 2, 2) where
            (0,0): TN, (0,1): FP, (1,0): FN, (1,1): TP
    """
    # Intersection is the true positives
    intersection = np.logical_and(preds, ground_truth)

    # Calculate confusion matrix for each class
    tp = np.sum(intersection, axis=0)
    fp = np.sum(preds, axis=0) - tp
    fn = np.sum(ground_truth, axis=0) - tp
    tn = ground_truth.shape[0] - tp - fp - fn

    # Confusion matrix shape is (#c, 2, 2)
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def confusion_matrix_from_images(preds, ground_truth, pred_threshold=0.0):
    """
    Calculate Confusion Matrix per class for entire batch of images.
    @param preds: model preds logits array,    (b, #c, h, w)
    @param ground_truth: one-hot ground truth, (b, #c, h, w)
    @param pred_threshold: Confidence threshold over which pixel prediction counted.
    @return: confusion matrix per class: shape (#c, 2, 2) where
            (0,0): TN, (0,1): FP, (1,0): FN, (1,1): TP
    """
    # Reduce all dims except classes dim (b,c,h,w) -> (b,h,w,c) -> (b*h*w, c)
    preds = reshape_channel_last(preds)
    ground_truth = reshape_channel_last(ground_truth)  # TODO: enforce bool type for gt?

    # Pred values are true if above threshold
    preds = preds > pred_threshold

    return confusion_matrix(preds, ground_truth)


def calculate_metrics(preds, label_masks, pred_threshold=0.0, zero_nans=True):
    """
    Calculate IoU, Precision and Recall per class for entire batch of images.
    @param preds: model preds logits array,   (b, #c, h, w)
    @param label_masks: one-hot ground truth, (b, #c, h, w)
    @param pred_threshold: Confidence threshold over which pixel prediction counted.
    @param zero_nans: whether to zero out nan metrics (true by default)
    @return: ious, precs, recall, acc per class: shape (#c)
    """
    CM = confusion_matrix_from_images(preds, label_masks, pred_threshold=pred_threshold)
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
    Creates a metrics dictionary, mapping metric_name --> metric_val.
    @param classes: Dictionary of class_name --> index in metric array
    @param loss: Either None, or if specified, PyTorch loss number for the epoch
    @param metrics: Each metric should be a numpy array of shape (num_classes)
    @return: Dictionary of metric_name --> metric_val.
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
