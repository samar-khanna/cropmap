import os
import re
import json
import random
import argparse
from collections import Counter, OrderedDict

from metrics import MeanMetric
from utils.colors import plot_color_legend
from eval.plotting import plot_hist, plot_images
from eval.utils import (
    parse_class_name_metric_type,
    calculate_metrics_from_confusion_matrix,
    get_per_class_confusion_matrix
)


def passed_arguments():
    parser = argparse.ArgumentParser(description=
                                     "Script to run evaluation for inference results.")
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
    parser.add_argument("--save_fig",
                        type=str,
                        default=None,
                        help="Path to dir where figures will be saved instead of displayed.")
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


def sort_key(file_name):
    d = re.search('[0-9]+', file_name)
    return int(file_name[d.start():d.end()]) if d else float('inf')


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


def json_dict_generator(paths):
    """
    Returns a generator over json objects using the paths to the .json files
    """
    for path in paths:
        with open(path, 'r') as f:
            dict_obj = json.load(f)
        yield dict_obj


def calculate_mean_results(metric_dicts):
    """
    Calculates mean metrics across all images in inference out.
    """
    # Calculate mean metrics across all images in inference out.
    mean_results = {}
    class_confusion_matrices = Counter()
    for i, metrics in enumerate(metric_dicts):

        class_cm = get_per_class_confusion_matrix(metrics)
        class_confusion_matrices.update(class_cm)

        for metric_name, val in metrics.items():
            class_name, metric_type = parse_class_name_metric_type(metric_name)

            if metric_type.lower() not in {"tn", "fp", "fn", "tp"}:
                if metric_name not in mean_results:
                    mean_results[metric_name] = MeanMetric(val)
                else:
                    mean_results[metric_name].update(val)

    # Get rid of MeanMetrics
    mean_results = {n: metric.item() for n, metric in mean_results.items()}

    # Add in the other metrics from the confusion matrix
    for class_name, cm in class_confusion_matrices.items():
        metric_types = ["iou", "prec", "recall"]
        cm_metrics = calculate_metrics_from_confusion_matrix(cm, metric_types=metric_types)
        for metric_type, val in cm_metrics.items():
            mean_results[f"{class_name}/{metric_type}"] = val

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
        class_name, metric_type = parse_class_name_metric_type(metric_name)

        if metric_type.find("class_count") > -1:
            if class_name.find("mean") == -1:  # don't log mean_class_count
                class_counts[class_name] = metric_val
            continue

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

    # Calculate mean of top k common classes for each metric type
    if class_counts:

        # If class was in class_counts, then get it from seen
        for class_name in class_counts:
            seen_results = seen.get(class_name)
            if seen_results is not None:
                classes_metrics[class_name] = seen_results
                del seen[class_name]

        topk_idx = min(topk-1, len(class_counts) - 1)
        desc_class_counts = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
        topk_class_counts = OrderedDict([(k, count) for k, count in desc_class_counts 
                                         if count >= desc_class_counts[topk_idx][1]])

        metric_types = classes_metrics["mean"].keys()
        topk_mean = {metric_type: MeanMetric() for metric_type in metric_types}

        for class_name, class_count in topk_class_counts.items():
            for metric_type in metric_types:
                topk_mean[metric_type].update(classes_metrics[class_name][metric_type])

        topk_mean = {metric_type: mean_result.item() for metric_type, mean_result in topk_mean.items()}
        classes_metrics[f"top_{topk}"] = topk_mean

        # Sort bar plot for top k classes in decreasing order of prevalance, 
        # Get rid of metrics outside of top k prevalence
        sorted_class_names = ["mean", f"top_{topk}"]
        sorted_class_names.extend(topk_class_counts.keys())
        classes_metrics = OrderedDict([(name, classes_metrics[name]) for name in sorted_class_names])

    return classes_metrics


if __name__ == "__main__":
    args = passed_arguments()

    with open(args.classes, 'r') as f:
        classes = json.load(f)

    inf_paths = args.inference
    assert len(inf_paths) > 0, "Need to specify at least one inf dir."

    if len(inf_paths) == 1:
        thresh = 0.2
        topk = 7
        inf_path = inf_paths[0]

        im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths = \
            get_inference_out_paths(inf_path)
        
        metric_dicts = json_dict_generator(metric_paths)

        # Get rid of MeanMetrics
        mean_results = calculate_mean_results(metric_dicts)

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
        if args.save_fig is not None or not args.text:
            # Mean result histogram
            hist_metrics = format_metrics_for_hist(mean_results, thresh=thresh, topk=topk)
            plot_hist(hist_metrics, thresh=thresh, topk=topk,
                      savefig=os.path.join(args.save_fig, "model_results.png"))

        # Plot grids of the images.
        if args.images:
            paths = zip(im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths)
            paths = list(paths)

            show_paths = random.sample(paths, 20)
            for ps in show_paths:
                plot_images(ps)

    # Plot comparative bar charts.
    else:
        topk = 4
        chosen_metric = "IoU"

        # Get results per inference set.
        inf_results = {}
        for inf_path in inf_paths:
            im_paths, gt_paths, gt_raw_paths, pred_paths, pred_raw_paths, metric_paths = \
                get_inference_out_paths(inf_path)
            
            metric_dicts = json_dict_generator(metric_paths)

            # Get rid of MeanMetrics
            mean_results = calculate_mean_results(metric_dicts)

            # Inference tag is descriptive name of expeirment
            inf_tag = inf_path.strip(os.path.sep).split(os.path.sep)[-1]
            print(f"Evaluation results for non-zero metrics in {inf_tag}:")
            for metric_name, metric_val in mean_results.items():
                if metric_val > 0.3:
                    print(f"{metric_name}: {round(metric_val, 3)}")

            inf_results[inf_tag] = format_metrics_for_hist(mean_results, thresh=0.01, topk=topk)

        combined_metrics = {}
        for inf_tag, metrics_for_hist in inf_results.items():
            for class_name, metrics in metrics_for_hist.items():
                chosen_metric_per_tag = combined_metrics.get(class_name.lower(), {})
                chosen_metric_per_tag[inf_tag] = metrics[chosen_metric.lower()]
                combined_metrics[class_name] = chosen_metric_per_tag

        print(combined_metrics)
        plot_hist(combined_metrics, topk=topk, ylabel=chosen_metric,
                  savefig=os.path.join(args.save_fig, "compare_models.png"))
