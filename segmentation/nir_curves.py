import torch
import torch.nn as nn
import numpy as np

import os
import json
import pickle
import argparse

from metrics import MeanMetric
from data_loaders.time_series_loader import TimeSeriesDataset


def passed_arguments():
    parser = argparse.ArgumentParser(description= \
                                         "Script to run inference for segmentation models.")
    parser.add_argument("-d", "--data_path",
                        type=str,
                        required=True,
                        help="Path to directory containing datasets.")
    parser.add_argument("--data_map",
                        type=str,
                        default=None,
                        help="Path to .json file with train/val/test split for experiment.")
    parser.add_argument("-s", "--set_type",
                        type=str,
                        default="val",
                        help="One of the train/val/test sets to perform inference.")
    parser.add_argument("--classes",
                        type=str,
                        default=os.path.join("segmentation", "classes.json"),
                        help="Path to .json index->class name file.")
    parser.add_argument("--inf_out",
                        type=str,
                        default=None,
                        help="Path to output directory to store inference results. " + \
                             "Defaults to `.../data_path/inference/`")
    parser.add_argument('-n', '--name',
                        type=str,
                        default=None,
                        help='Experiment name, used as directory name in inf_dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = passed_arguments()

    # Classes is dict of {class_name --> class_id}
    with open(args.classes, 'r') as f:
        classes = json.load(f)

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cpu")

    # interest_classes = [
    #     "Corn", "Soybeans", "Potatoes", "Oats", "Apples", "Fallow-Idle Cropland",
    #     "Almonds", "Pistachios", "Cotton", "Winter Wheat", "Grapes", "Alfalfa", "Walnuts"
    # ]
    dataset = TimeSeriesDataset(
        args.data_path,
        classes,
        data_map_path=args.data_map,
        inf_mode=True
    )
    train_loader, val_loader, test_loader = dataset.create_data_loaders(batch_size=8)

    months = {0: "01-01", 1: "01-17", 2: "02-02", 3: "02-18", 4: "03-06", 5: "03-22",
              6: "04-07", 7: "04-23", 8: "05-09", 9: "05-25", 10: "06-10", 11: "06-26",
              12: "07-12", 13: "07-28", 14: "08-13", 15: "08-29", 16: "09-14", 17: "09-30",
              18: "10-16", 19: "11-01", 20: "11-17", 21: "12-03", 22: "12-19"}
    class_avg_nir = {class_id: {month: MeanMetric() for month in months.values()}
                     for class_id in range(len(classes))}
    class_avg_swir1 = {class_id: {month: MeanMetric() for month in months.values()}
                       for class_id in range(len(classes))}
    class_avg_swir2 = {class_id: {month: MeanMetric() for month in months.values()}
                       for class_id in range(len(classes))}

    for batch_index, (xs, y) in enumerate(val_loader):
        xs, y = dataset.shift_sample_to_device((xs, y), device)

        y = y.permute(1, 0, 2, 3).numpy()  # shape (c, b, h, w)

        for i, x in enumerate(xs):
            m = months[i]

            x = x.numpy()
            valid_mask = np.any(x != 0., axis=1)  # shape (b, h, w)

            nir_band = x[:, 4, :, :]  # shape (b, h, w)
            swir1_band = x[:, 5, :, :]  # shape (b, h, w)
            swir2_band = x[:, 6, :, :]  # shape (b, h, w)

            # Get vectors per interest class
            for class_id, class_mask in enumerate(y):
                combined_mask = valid_mask & class_mask.astype(np.bool)  # shape (b, h, w)

                for band, class_avg in zip([nir_band, swir1_band, swir2_band],
                                           [class_avg_nir, class_avg_swir1, class_avg_swir2]):

                    features = band[combined_mask]  # shape (m,), m is num correct class pixels
                    if len(features) > 0:
                        class_avg[class_id][m].update(np.mean(features))

    rev_classes = {og_id: class_name for class_name, og_id in classes.items()}
    final_dict = {'nir': {}, 'swir1': {}, 'swir2': {}}
    for band_name, class_avg_dict in [('nir', class_avg_nir),
                                      ('swir1', class_avg_swir1),
                                      ('swir2', class_avg_swir2)]:
        for class_id, month_dict in class_avg_dict.items():
            for m, class_avg in month_dict.items():
                try:
                    month_dict[m] = class_avg.item()
                except ValueError:
                    month_dict[m] = -1
            final_dict[band_name][rev_classes[dataset.map_idx_to_class[class_id]]] = month_dict

    with open(os.path.join(args.inf_out, f'{args.name}_growth_curves.pkl'), 'wb') as f:
        pickle.dump(final_dict, f)

    print('Done!')
