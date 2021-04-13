import os
import json
import argparse
import numpy as np
from collections import defaultdict

from eval.plotting import plot_shot_curves
from eval.utils import (
    get_per_class_confusion_matrix,
    calculate_metrics_from_confusion_matrix
)


def passed_arguments():
    parser = argparse.ArgumentParser(description="Script to plot meta-inf shot curves.")
    parser.add_argument("--files",
                        nargs='+',
                        type=str,
                        required=True,
                        help="Path(s) to .json files containing shot curve data")
    parser.add_argument("--plot_classes",
                        nargs='+',
                        type=str,
                        default=['mean'],
                        help="Names of classes for which to plot shot curves.")
    parser.add_argument("--save_fig",
                        type=str,
                        default=None,
                        help="Path to dir where figures will be saved instead of displayed.")
    parser.add_argument("--classes",
                        type=str,
                        default=os.path.join("segmentation", "classes.json"),
                        help="Path to .json class->index name file.")
    args = parser.parse_args()
    return args


def get_class_to_shots_metrics(curve_data):
    # Get confusion matrices per class
    # {shot_num: {class_name: [cm_trial1, cm_trial2, ...], ...}, ...}
    class_to_shots = defaultdict(lambda: defaultdict(list))
    for shot_num, trial_results in curve_data.items():

        for trial_num, metric_dict in enumerate(trial_results):
            class_cms = get_per_class_confusion_matrix(metric_dict)

            for class_name, cm in class_cms.items():
                # TODO: Keep general CM? Or get prec and recall?
                metrics = calculate_metrics_from_confusion_matrix(cm, ("iou",))
                if 'mean' in class_name.lower():
                    print(shot_num, metrics['iou'])
                class_to_shots[class_name][shot_num].append(metrics["iou"])

    return class_to_shots


def get_class_to_shots_avg(class_to_shots_metrics):
    class_to_shots_avg = defaultdict(dict)
    for class_name, shot_dict in class_to_shots_metrics.items():
        for shot_num, metric_list in shot_dict.items():
            cm_mean = np.mean(np.array(metric_list), axis=0)
            cm_stddev = np.std(np.array(metric_list), axis=0)
            class_to_shots_avg[class_name][shot_num] = (cm_mean, cm_stddev)

    return class_to_shots_avg


if __name__ == "__main__":
    args = passed_arguments()

    assert len(args.files) > 0, "Require paths to .json files containing shot curves"

    experiment_dict = defaultdict(dict)
    for file in args.files:
        exp_name = file.split(os.path.sep)[-1].replace(".json", "").replace("shot_curve", "")
        with open(file, 'r') as f:
            curve_data = json.load(f)

        class_to_shots_metrics = get_class_to_shots_metrics(curve_data)

        class_to_shots_avg = get_class_to_shots_avg(class_to_shots_metrics)

        for plot_class in args.plot_classes:
            experiment_dict[exp_name][plot_class.lower()] = class_to_shots_avg[plot_class]

    save_file = os.path.join(args.save_fig, 'shot_curves.png') if args.save_fig else None
    plot_shot_curves(experiment_dict, savefig=save_file)

    pass
