import numpy as np


def parse_class_name_metric_type(string):
    """
    Parses key in input metrics dictionary into class name and metric type for that class.
    """
    class_name = string.split('/')[0].lower()  # Important to lowercase
    metric_type = string.split('/')[-1].lower()
    class_name = class_name.replace("class_", "")
    return class_name, metric_type


def calculate_metrics_from_confusion_matrix(cm, metric_types=("iou", "prec", "recall")):
    """
    Calculates metrics based on confusion matrix.
    Requires:
    `confusion_matrix`: (2, 2) array with [[tn, fp], [fn, tp]]
    """
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    metrics = {}
    for metric_type in metric_types:
        if metric_type.lower() == "iou":
            iou = tp / (tp + fp + fn)
            iou = 0. if np.isnan(iou) else iou
            metrics["iou"] = iou
        elif metric_type.lower() == "prec":
            prec = tp / (tp + fp)
            prec = 0. if np.isnan(prec) else prec
            metrics["prec"] = prec
        elif metric_type.lower() == "recall":
            recall = tp / (tp + fn)
            recall = 0. if np.isnan(recall) else recall
            metrics["recall"] = recall

    return metrics


def get_per_class_confusion_matrix(metric_dict):
    """
    Creates a confusion matrix for each class in metric_dict using
    tp, fp, fn, tn. Return dictionary is empty if these aren't present
    @param metric_dict: Dictionary of {metric_name: metric_value}
    @return:
    """
    class_confusion_matrices = {}
    for metric_name, val in metric_dict.items():
        class_name, metric_type = parse_class_name_metric_type(metric_name)

        if metric_type.lower() in {"tn", "fp", "fn", "tp"}:
            cm = class_confusion_matrices.get(class_name, np.zeros((2, 2)))
            if metric_type.lower() == "tn":
                cm[0, 0] += val
            elif metric_type.lower() == "fp":
                cm[0, 1] += val
            elif metric_type.lower() == "fn":
                cm[1, 0] += val
            elif metric_type.lower() == "tp":
                cm[1, 1] += val
            class_confusion_matrices[class_name] = cm

    return class_confusion_matrices


def classes_dist(path_to_gt, classes):
    """
    Records the distribution of classes in a given ground truth mask.
    Requires:
      `path_to_gt`: path to a ground truth `.tif` file representing the mask.
    Returns:
      A dictionary of class_name --> count
    """
    import rasterio
    with rasterio.open(path_to_gt) as m:
        mask = m.read()

    unique, counts = np.unique(mask, return_counts=True)
    index_dist = dict(zip(unique, counts))

    invert_classes = {ind: name for name, ind in classes.items()}

    dist = {invert_classes[ind]: int(count)
            for ind, count in index_dist.items() if ind > 0}

    return dist