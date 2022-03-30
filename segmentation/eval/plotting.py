import json

import numpy as np
from PIL import Image
from collections import defaultdict
from matplotlib import pyplot as plt


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


def plot_shot_curves(experiment_dict, savefig=None):
    # Invert so that now it is class -> experiments
    class_to_experiment = defaultdict(dict)
    for exp_name, class_to_shots_avg in experiment_dict.items():
        for class_name, shot_dict in class_to_shots_avg.items():
            class_to_experiment[class_name][exp_name] = shot_dict

    for class_name, experiment_results in class_to_experiment.items():
        fig, ax = plt.subplots()
        for exp_name, shot_results in experiment_results.items():
            x = [int(k) for k in shot_results.keys()]
            y = [mean for mean, std in shot_results.values()]
            yerr = [std/np.sqrt(10) for mean, std in shot_results.values()]

            # ax.plot(x, y, label=exp_name)
            ax.errorbar(x, y, yerr=yerr, fmt='x-', label=exp_name)

        ax.set_ylabel("IoU")
        ax.set_xlabel("Number of Input Shots")
        ax.set_title(f"{class_name} class results")
        # ax.set_xticks(x)
        # ax.set_xticklabels(x_labels)
        ax.legend(fontsize=8, framealpha=0.4)

        if not savefig:
            fig.tight_layout()
            fig.show()
            plt.show()
        else:
            fig.set_size_inches(12, 6)
            fig.savefig(savefig)
            plt.close()

    pass


def plot_growth_curves(band_results, interest_classes, title="NIR band results"):
    fig, ax = plt.subplots()
    for c in interest_classes:
        class_results = band_results[c]

        xticks = class_results.keys()
        x = list(range(len(class_results)))
        y = class_results.values()

        ax.set_xticks(x)
        ax.set_xticklabels(xticks, fontsize=8)
        ax.plot(x, y, 'x-', label=c)

    ax.legend(fontsize=8, framealpha=0.4)
    ax.set_ylabel("Band raw value")
    ax.set_xlabel("Time in year")
    ax.set_title(title)

    plt.show()
    return


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
    # sort_keys = {"mean": 0, f"top_{topk}": 1, "corn": 2, "soybeans": 3}
    # sort_key = lambda k: sort_keys.get(k.lower(), len(sort_keys))
    x_labels = metrics.keys()
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
    ax.legend(fontsize=8, framealpha=0.4)

    if not savefig:
        fig.tight_layout()
        fig.show()
        plt.show()
    else:
        fig.set_size_inches(12, 6)
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
