import os
import re
import json
import random
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from segmentation.metrics import MeanMetric
from segmentation.utils.colors import plot_color_legend


def passed_arguments():
    parser = argparse.ArgumentParser(description= \
                                         "Script to run inference for segmentation models.")
    parser.add_argument("-inf", "--inference",
                        nargs="+",
                        type=str,
                        required=True,
                        help="Path to directory(ies) containing inference results.")
    parser.add_argument("--text",
                        action="store_true",
                        default=False,
                        help="Only print the text results from evaluation.")
    parser.add_argument("--images",
                        action="store_true",
                        default=False,
                        help="Show a sequence of 20 images from inference.")
    parser.add_argument("--colors",
                        action="store_true",
                        default=False,
                        help="Plot the color legend for the classes.")
    parser.add_argument("--classes",
                        type=str,
                        default=os.path.join("segmentation", "classes.json"),
                        help="Path to .json class->index name file.")
    args = parser.parse_args()
    return args


def get_inference_out_paths(inf_path):
    """
    Get the paths to each type of file in inf out.
    (im_path, gt_path, gt_raw_paths, pred_path, pred_raw_path, metric_path)
    """
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
    return im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths


def calculate_mean_results(metric_paths):
    """
    Calculates mean metrics across all images in inference out.
    """
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
    return mean_results


def format_metrics_for_hist(metrics, thresh=0.2, topk=5):
    """
    Creates metrics dictionary ready for histogram plotting
    `metrics`: map of metric name -> metric val
    `topk`: if class counts available, also computes mean of top k most common classes
    `thresh`: percentage threshold for iou, prec, recall
    Returns:
        metrics: Dict ordered as follows: \n
                {class1_name: {type1: val, type2: val, ...},
                 class2_name: {type1: val, type2: val, ...},
                 ...}
    """
    class_counts = {}
    classes_metrics = {}
    seen = {}
    for metric_name, metric_val in metrics.items():
        class_name = metric_name.split('/')[0]
        metric_type = metric_name.split('/')[-1]
        class_name = class_name.replace("class_", "")

        if metric_type.find("class_count") > -1:
            class_counts[class_name] = metric_val
            continue

        # Store metric results only for mean metrics and metrics above threshold
        class_results = classes_metrics.get(class_name, OrderedDict())
        class_results[metric_type] = metric_val
        classes_metrics[class_name] = class_results
        # if class_results or class_name.find("mean") > -1 or metric_val > thresh:
        #     if class_name in seen:
        #         seen_results = seen[class_name]
        #         class_results.update(seen_results)
        #         del seen[class_name]
        #
        #     class_results[metric_type] = metric_val
        #     classes_metrics[class_name] = class_results
        # else:
        #     seen_results = seen.get(class_name, OrderedDict())
        #     seen_results[metric_type] = metric_val
        #     seen[class_name] = seen_results

    # Calculate mean of top k common classes for each metric type
    if class_counts:
        # TODO: Get rid of this nonsense
        # unseen_classes = set()
        # for class_name in class_counts:
        #     if class_name not in classes_metrics:
        #         unseen_classes.add(class_name)
        # print(unseen_classes)
        # for class_name in unseen_classes:
        #     del class_counts[class_name]

        topk = min(topk, len(class_counts) - 1)
        sorted_counts = sorted(class_counts.values())
        topk_class_counts = dict(filter(lambda x: x[1] > sorted_counts[topk], class_counts.items()))

        metric_types = classes_metrics["mean"].keys()
        topk_mean = {metric_type: MeanMetric() for metric_type in metric_types}

        for class_name, class_count in topk_class_counts.items():
            for metric_type in metric_types:
                topk_mean[metric_type].update(classes_metrics[class_name][metric_type])

        topk_mean = {metric_type: mean_result.item() for metric_type, mean_result in topk_mean.items()}
        classes_metrics[f"top_{topk}"] = topk_mean

        # also get rid of displaying all metrics that don't lie in top k
        non_class_names = {"mean", f"top_{topk}"}
        filter_f = lambda x: x[0].lower() in non_class_names or x[0].lower() in topk_class_counts
        classes_metrics = dict(filter(filter_f, classes_metrics))

    return classes_metrics


def sort_key(file_name):
    d = re.search('[0-9]+', file_name)
    return int(file_name[d.start():d.end()]) if d else float('inf')


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

    dist = {invert_classes[ind]: int(count) \
            for ind, count in index_dist.items()}

    return dist


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


def plot_hist(metrics, thresh=0.01, topk=5,
              ylabel="Metric Scores", title="Metric scores by class", savefig=None):
    """
    Plots a histogram on the mean results of the inference task.
    Requires:
      `metrics`: Dict ordered as follows: \n
                `{class1_name: {type1: val, type2: val, ...},
                  class2_name: {type1: val, type2: val, ...},
                  ...}
    """
    # Keep mean results first.
    sort_keys = {"mean": 0, f"top_{topk}": 1, "corn": 2, "soybeans": 3}
    sort_key = lambda k: sort_keys.get(k.lower(), len(sort_keys))
    x_labels = sorted(metrics.keys(), key=sort_key)
    x = np.arange(len(x_labels))

    # List of rectangles for each metric type
    metric_type_results = {}
    for i, label in enumerate(x_labels):
        class_results = metrics[label]

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
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    if not savefig:
        fig.tight_layout()
        fig.show()
        plt.show()
    else:
        fig.savefig(savefig)
        plt.close()


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

    with open(args.classes, 'r') as f:
        classes = json.load(f)

    inf_paths = args.inference
    assert len(inf_paths) > 0, "Need to specify at least one inf dir."

    if len(inf_paths) == 1:
        inf_path = inf_paths[0]

        im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths = \
            get_inference_out_paths(inf_path)

        # Get rid of MeanMetrics
        mean_results = calculate_mean_results(metric_paths)

        print("Evaluation results for non-zero metrics:")
        for metric_name, metric_val in mean_results.items():
            if metric_val > 0:
                print(f"{metric_name}: {round(metric_val, 3)}")

        # Plot color map if config file given.
        if args.colors:
            print("Warning: Showing color map without config file.")
            interest_classes = sorted(classes.keys(), key=classes.get)
            remapped_classes = {}
            for i, class_name in enumerate(interest_classes):
                remapped_classes[class_name] = i

            plot_color_legend(remapped_classes)

        ## Start plotting results.
        if not args.text:
            # Mean result histogram
            hist_metrics = format_metrics_for_hist(mean_results, thresh=0.2, topk=5)
            plot_hist(hist_metrics, thresh=0.2, topk=5)

        # Plot grids of the images.
        if args.images:
            paths = zip(im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths)
            paths = list(paths)

            show_paths = random.sample(paths, 20)
            for ps in show_paths:
                plot_images(ps)

    # Plot comparative bar charts.
    else:
        # Get results per inference set.
        inf_results = {}
        for inf_path in inf_paths:
            im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths = \
                get_inference_out_paths(inf_path)

            # Get rid of MeanMetrics
            mean_results = calculate_mean_results(metric_paths)
            print("Evaluation results for non-zero metrics:")
            for metric_name, metric_val in mean_results.items():
                if metric_val > 0.3:
                    print(f"{metric_name}: {round(metric_val, 3)}")

            # Inference tag is descriptive name of expeirment
            inf_tag = inf_path #os.path.split(inf_path)[-1]
            inf_results[inf_tag] = format_metrics_for_hist(mean_results, thresh=0.01, topk=4)

        topk = 4
        chosen_metric = "IoU"
        combined_metrics = {}
        for inf_tag, metrics_for_hist in inf_results.items():
            for class_name, metrics in metrics_for_hist.items():
                chosen_metric_per_tag = combined_metrics.get(class_name.lower(), {})
                chosen_metric_per_tag[inf_tag] = metrics[chosen_metric]
                combined_metrics[class_name] = chosen_metric_per_tag

        plot_hist(combined_metrics, topk=topk, ylabel=chosen_metric, savefig="/home/sak296/fig1.png")
