import torch
import torch.nn as nn
import numpy as np

import os
import json
import time
import pickle
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.stats import mode
from sklearn.metrics import pairwise_distances
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tslearn.metrics import cdist_dtw

from metrics import MeanMetric
from utils.loading import load_model
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
    parser.add_argument("--inf_out",
                        type=str,
                        required=True,
                        help="Path to output directory to store inference results. ")
    parser.add_argument("--model_config",
                        type=str,
                        default=None,
                        help="Path to SSAVF model config")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="Path to SSAVF model.")
    parser.add_argument("-s", "--set_type",
                        type=str,
                        default="val",
                        help="One of the train/val/test sets to perform inference.")
    parser.add_argument("--classes",
                        type=str,
                        default=os.path.join("segmentation", "classes.json"),
                        help="Path to .json index->class name file.")
    args = parser.parse_args()
    return args


def get_pixel_dataset(loader, device):
    # Assuming batch size is fixed to 1
    X, Y = [], []
    for batch_idx, (xs, y) in enumerate(loader):
        xs, y = dataset.shift_sample_to_device((xs, y), device)  # [x1, ..., xt], x_i: (1, c, h, w)

        t = len(xs)
        b, c, h, w = xs[0].shape
        xs = torch.stack(xs).permute(1, 3, 4, 0, 2)  # (t, b, c, h, w) -> (b, h, w, t, c)
        xs = xs.reshape(-1, t, c)  # (b, h, w, t, c) -> (b*h*w, t, c)
        y = y.reshape(-1)  # (b, 1, h, w) -> (b*h*w)

        X.append(xs)
        Y.append(y)

    X = torch.cat(X, dim=0)  # (N*h*w, t, c)
    Y = torch.cat(Y)  # (N*h*w)

    return X.numpy(), Y.numpy()


def get_random_samples(X, Y, num_per_class=3):
    sample_X, sample_Y = [], []
    classes, counts = np.unique(Y, return_counts=True)
    print(f"Train classes {classes}, counts: {counts}")
    for c in classes[counts >= num_per_class]:
        class_inds, = np.nonzero(Y == c)
        sample_inds = np.random.choice(class_inds, size=num_per_class, replace=False)

        sample_X.append(X[sample_inds, ...])
        sample_Y.append(Y[sample_inds, ...])

    return np.concatenate(sample_X, axis=0), np.concatenate(sample_Y, axis=0)


def get_avg_samples(X, Y, num_per_class=1):
    sample_X, sample_Y = [], []
    classes, counts = np.unique(Y, return_counts=True)
    for c in classes[counts >= num_per_class]:
        class_inds, = np.nonzero(Y == c)
        class_samples = X[class_inds, ...]

        # Randomly partition the samples of a class into num_per_class chunks
        np.random.shuffle(class_samples)
        splits = np.array_split(class_samples, num_per_class, axis=0)
        for chunk in splits:
            sample_vec = np.mean(chunk, axis=0)  # (t, c)

            sample_X.append(sample_vec.reshape(1, *X.shape[1:]))  # (1, t, c)
            sample_Y.append(np.array([c]))

    return np.concatenate(sample_X, axis=0), np.concatenate(sample_Y, axis=0)


def ssavf_single_dist(model, x_tr, x_val, chunk_size=100000):
    dist_row = []

    num_split = len(x_val)//chunk_size if len(x_val)//chunk_size > 0 else 1
    for x_va in np.array_split(x_val, num_split, axis=0):
        xt = torch.from_numpy(x_tr)  # (t, c)
        xv = torch.from_numpy(x_va)  # (chunk, t, c)

        xt = xt.expand(xv.shape[0], -1, -1)  # (chunk, t, c)

        # Need in (chunk, c, t) shape to input to model
        x_aligned, x_shift, theta_pred = model(xt.permute(0, 2, 1), xv.permute(0, 2, 1))

        row = ((x_aligned - x_shift) ** 2).sum(-1).sum(-1)  # (chunk,)
        dist_row.extend(row.detach().numpy())

    return np.array(dist_row)  # (N_val,)


def ssavf_dist(model, x_train, x_val, chunk_size=100000):
    model.eval()

    sim_mat = np.zeros((len(x_train), len(x_val)))
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as exec:
        future_to_tr_idx = {
            exec.submit(ssavf_single_dist, model, x_train[i], x_val, chunk_size): i
            for i in range(len(x_train))
        }

        for future in as_completed(future_to_tr_idx):
            idx = future_to_tr_idx[future]

            dist_row = future.result()
            sim_mat[idx] = dist_row

    return sim_mat

    # for x_tr in x_train:
    #     x_tr = torch.from_numpy(x_tr)  # (t, c)
    #
    #     dist_row = []
    #
    #     for batch_idx, (xs, y) in enumerate(val_loader):
    #         x_v, y = dataset.shift_sample_to_device((xs, y), device)  # [x1,...,xt], x_i: (1,c,h,w)
    #         h, w = x_v[0].shape[-2:]
    #
    #         # [x1,...,xt], x_i: (1,c,h,w)
    #         x_t = [t.expand(h, w, -1).permute(2, 0, 1).unsqueeze(0) for t in x_tr]
    #
    #         x_aligned, x_shift, theta_pred = model(x_t, x_v)
    #
    #         # Append a row of length
    #         row = ((x_aligned - x_shift)**2).sum(-1).sum(-1)  # (h*w,)
    #         dist_row.extend(row.detach().numpy())
    #
    #     sim_mat.append(np.array(dist_row))
    #
    # return sim_mat


def knn_accs(nn_results, y_train, y_test, k_neigh=(1, 3, 5, 11)):
    accs = {}
    for k in k_neigh:
        neighbours = nn_results[:k, :]
        preds = y_train[neighbours]  # (k, n_test)
        preds = mode(preds, axis=0)[0].reshape(-1)  # (n_test)

        known_mask = y_test != -1
        preds = preds[known_mask]
        gt = y_test[known_mask]

        accs[k] = np.sum(preds == gt) / gt.shape[0]

    return accs


if __name__ == "__main__":
    args = passed_arguments()
    os.makedirs(args.inf_out, exist_ok=True)

    model = None
    if args.checkpoint is not None and args.model_config is not None:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
        model = load_model(model_config, 1, from_checkpoint=args.checkpoint)

    print("Starting KNN accuracy test...")
    print(f"Using {os.cpu_count()} CPUs")

    # Classes is dict of {class_name --> class_id}
    with open(args.classes, 'r') as f:
        classes = json.load(f)

    interest_classes = ["Corn", "Soybeans", "Rice", "Alfalfa", "Grapes", "Almonds", "Pecans",
                        "Peanuts", "Pistachios", "Walnuts", "Potatoes", "Oats", "Apples",
                        "Cotton", "Dry Beans", "Sugarbeets", "Winter Wheat", "Spring Wheat",
                        "Pop or Orn Corn", "Other Hay-Non Alfalfa", "Grass-Pasture", "Woody Wetlands",
                        "Herbaceous Wetlands", "Developed-Open Space", "Deciduous Forest",
                        "Fallow-Idle Cropland"]

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cpu")

    dataset = TimeSeriesDataset(
        args.data_path,
        classes,
        interest_classes=interest_classes,
        data_map_path=args.data_map,
        inf_mode=True,
        use_one_hot=False,
        transforms={
            "MaskCloudyTargetsTransform":
                {"mask_value": -1, "cloud_value": 0.0, "is_conservative": False}
        }
    )
    train_loader, val_loader, test_loader = dataset.create_data_loaders(batch_size=1)

    # Assuming batch size is fixed to 1
    X_train_full, y_train_full = get_pixel_dataset(train_loader, device)
    X_val_full, y_val_full = get_pixel_dataset(val_loader, device)
    Ntr, t, c = X_train_full.shape
    Nval, _, _ = X_val_full.shape

    print(X_train_full.shape, y_train_full.shape)
    print(X_val_full.shape, y_val_full.shape)
    print("Starting trials")

    exp_dict = {method: defaultdict(lambda: defaultdict(list))
                for method in ["dtw", "euc", "ssavf"]}

    ## ============================================================================
    # DO FULL MATRIX TEST:
    # X_train = X_train_full
    # y_train = y_train_full
    # _, t, c = X_train.shape
    # X_train = X_train.reshape(-1, t * c)
    # X_val = X_val.reshape(-1, t * c)
    #
    # all_accs = defaultdict(list)
    # total_time = 0
    # for x_val in np.array_split(X_val, 8):
    #     stt = time.time()
    #     euc_sim_mat = pairwise_distances(X_train, x_val, n_jobs=-1)  # (n_train, 8)
    #     total_time += time.time() - stt
    #
    #     euc_nns = np.argsort(euc_sim_mat, axis=0)
    #
    #     # Try 1, 3, 5 NNs
    #     euc_accs = knn_accs(euc_nns, y_train, y_val, k_neigh=(1, 3, 5, 11))
    #     for k, acc in euc_accs.items():
    #         all_accs[k].append(acc)
    #
    # for k, acc_list in all_accs.items():
    #     acc = np.mean(acc_list)
    #     exp_dict['euc'][X_train.shape[0]][k].append(acc)
    #     print(f"{X_train.shape[0]} sample Euclidean KNN {k} neighbours accuracy: {acc}")
    #
    # print(f"Took {total_time}s to finish Euclidean comp")
    #
    # with open(os.path.join(args.inf_out, 'knn_euc_full_jun-jul_usa_classes.json'), 'w') as f:
    #     json.dump(exp_dict, f, indent=1)
    ## ============================================================================

    for trial in range(5):
        ## USE SAMPLE OF VAL FOR DTW FEASABILITY
        val_subset_inds = np.random.choice(len(X_val_full), 224 * 224, replace=False)
        X_val, y_val = X_val_full[val_subset_inds, ...], y_val_full[val_subset_inds]

        # X_val, y_val = X_val_full, y_val_full

        # for n in [1, 2, 3, 5]:
        for n in [1, 2, 5, 11]:
            # X_train, y_train = get_random_samples(X_train_full, y_train_full, num_per_class=n)
            X_train, y_train = get_avg_samples(X_train_full, y_train_full, num_per_class=n)

            # ============================
            # DTW
            # ============================
            # # Sort training points in increasing order of dist
            # stt = time.time()
            # dtw_sim_mat = cdist_dtw(X_train, X_val, n_jobs=-1,  # (n_train, n_test)
            #                         global_constraint="sakoe_chiba", sakoe_chiba_radius=round(0.75*t))
            # print(f"Took {time.time() - stt}s to finish DTW comp")
            # dtw_nns = np.argsort(dtw_sim_mat, axis=0)
            #
            # # Try 1, 3, 5 NNs
            # dtw_accs = knn_accs(dtw_nns, y_train, y_val, k_neigh=(1, 3, 5, 11))
            # for k, acc in dtw_accs.items():
            #     exp_dict['dtw'][n][k].append(acc)
            #     print(f"{n} sample DTW KNN {k} neighbours accuracy: {acc}")

            # ============================
            # SSAVF
            # ============================
            if model is not None:
                stt = time.time()
                ssavf_sim_mat = ssavf_dist(model, X_train, X_val)
                print(f"Took {time.time() - stt}s to finish SSAVF comp")
                ssavf_nns = np.argsort(ssavf_sim_mat, axis=0)

                # Try 1, 3, 5 NNs
                ssavf_accs = knn_accs(ssavf_nns, y_train, y_val, k_neigh=(1, 3, 5, 11))
                for k, acc in ssavf_accs.items():
                    exp_dict['ssavf'][n][k].append(acc)
                    print(f"{n} sample SSAVF KNN {k} neighbours accuracy: {acc}")

            # ============================
            # EUC
            # ============================
            # Reshape for euclidean distance comp
            _, t, c = X_train.shape
            stt = time.time()
            euc_sim_mat = pairwise_distances(
                X_train.reshape(-1, t*c), X_val.reshape(-1, t*c), n_jobs=-1
            )  # (n_train, n_test)
            print(f"Took {time.time() - stt}s to finish Euclidean comp")
            euc_nns = np.argsort(euc_sim_mat, axis=0)

            # Try 1, 3, 5 NNs
            euc_accs = knn_accs(euc_nns, y_train, y_val, k_neigh=(1, 3, 5, 11))
            for k, acc in euc_accs.items():
                exp_dict['euc'][n][k].append(acc)
                print(f"{n} sample Euclidean KNN {k} neighbours accuracy: {acc}")

            with open(os.path.join(args.inf_out, 'knn_ssavf_euc_avg_usa_classes.json'), 'w') as f:
                json.dump(exp_dict, f, indent=1)
