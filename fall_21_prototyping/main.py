import rasterio
from sklearn import neighbors, linear_model, metrics, neural_network ##
from tslearn.metrics import dtw, soft_dtw
from tslearn.barycenters import dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter
import os
import numpy as np
import pickle
from collections import Counter
from interest_classes import interest_classes
import random
from pprint import pprint
samples_per_class = 5
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import torch
from copy import copy

class MaskCloudyTargetsTransform:
    def __init__(self, mask_value=-1, cloud_value=0., is_conservative=True):
        """
        Replaces those pixels in target to mask_value which correspond with cloudy
        pixels in the input. If conservative, then takes the union of cloudy pixels
        in time series input. If not conservative, takes the intersection.
        @param mask_value: Value with which to replace target cloudy pixels.
        @param is_conservative: Whether to use logical OR vs AND to make mask of cloudy pixels.
        """
        self.mask_value = mask_value
        self.cloud_value = cloud_value
        self.is_conservative = is_conservative

    def process(self, xs, y):

        invalid_mask = np.zeros(xs[0].shape[1:], dtype=np.bool)  # shape (h, w)
        for t, x_t in enumerate(xs):

            invalid = ~np.any(x_t != self.cloud_value, axis=0)  # shape (h, w)
            if t == 0 or self.is_conservative:
                invalid_mask = invalid_mask | invalid
            else:
                invalid_mask = invalid_mask & invalid
        y[0][invalid_mask] = self.mask_value  # shape (1, h, w)
        return xs, y
mask_cloud_transform = MaskCloudyTargetsTransform()

basedir = "/share/bharath/sak296/data_usa_2017/"
save_basedir = "/share/bharath/bw462/sat/knn_caching/"

regions = os.listdir(basedir)
all_dirs = []
for region in regions:
    region_basedir = f"{basedir}/{region}"
    region_dirs = [f"{region_basedir}/{d}/2017/" for d in os.listdir(region_basedir)]
    all_dirs.append(region_dirs)
print(all_dirs)

# dates are the same
start_month = 4
end_month = 7
dates = [d for d in os.listdir(all_dirs[0][0]) if os.path.isdir(f"{all_dirs[0][0]}/{d}") and int(d.split('-')[0]) in range(start_month, end_month+1)]
dates = sorted(dates)
print("Dates for data:", dates)

values = []
targets = []
for region, dir_group in zip(regions, all_dirs):
    sub_values = [[], []]
    sub_targets = [[], []]
    for data_dir in dir_group:
        # Use fact that  the subname (e.g. cali) is different then full name
        tile_idx = int(data_dir.split('loc')[1].split('/')[0])
        split = 'train' if tile_idx%3 else 'test'
        split_idx = 0 if split=='train' else 1
        save_dir = data_dir.replace(basedir, save_basedir).replace(region, '_'.join([region, split]))
        save_value_path = save_dir + f"values_{start_month}_{end_month}.pickle"
        save_target_path = save_dir + f"targets_{start_month}_{end_month}.pickle"
        if os.path.exists(save_value_path) and os.path.exists(save_target_path):
            print(f"Loading from {save_dir}")
            x = pickle.load(open(save_value_path, 'rb'))
            y = pickle.load(open(save_target_path, 'rb'))
        else:
            print(f"Generating {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            region_data_stack = []
            for date in dates:
                image_path = f"{data_dir}/{date}/mosaic.tif"
                with rasterio.open(image_path, 'r') as image_reader:
                    image = image_reader.read()
                region_data_stack.append(image)
            region_data_stack = np.stack(region_data_stack)
            with rasterio.open(f"{data_dir}/ground_truth.tif", 'r') as reader:
                region_target_image = np.array(reader.read(), dtype=float)
            region_data_stack, region_target_image = mask_cloud_transform.process(region_data_stack, region_target_image)
            x = region_data_stack.reshape(region_data_stack.shape[0], region_data_stack.shape[1], -1).squeeze()
            y = region_target_image.reshape(-1).squeeze()
            clear_idx = np.argwhere(y!=-1)
            y = y[clear_idx].squeeze(axis=-1)
            x = np.take(x, clear_idx, axis=2).squeeze(axis=3)

            interest_idx = np.argwhere(np.isin(y, interest_classes))
            y = y[interest_idx].squeeze(axis=-1)
            x = np.take(x, interest_idx, axis=2).squeeze(axis=3)

            pickle.dump(x, open(save_value_path, "wb"))
            pickle.dump(y, open(save_target_path, "wb"))
        if not len(y): continue
        print("Appending", save_dir)
        sub_values[split_idx].append(x)
        sub_targets[split_idx].append(y)
    # print(len(sub_values), len(sub_values[0]))
    values.append(copy(sub_values))
    targets.append(copy(sub_targets))


generalization_region = sys.argv[1] # 'us_south' # 'washington'
generalization_region_i = regions.index(generalization_region)


# values and targets are still of length 5 but have train and test components to them
#
# data settings are "generalization", "direct" and "global_to_single"
# generalization = train on A-D, generalize to E (test split)
# direct train/test on splits of E
# global_to_single = fusion of two above train sets
processed_values = [[], []]
processed_targets_with_region_is = [[], []]
data_mode = "global_to_single"
# data_mode = "global_to_single"
for region_i, (x_data, y_data) in enumerate(zip(values, targets)):
    for orig_split_idx in [0, 1]: # train and test
        # operate on region_i and used_split_idx
        if data_mode == 'generalization':
            if orig_split_idx==0 and region_i==generalization_region_i: continue
            used_split_idx = 0 if region_i!=generalization_region_i else orig_split_idx # so 1
        elif data_mode == 'direct':
            if region_i != generalization_region_i: continue
            used_split_idx = orig_split_idx
        elif data_mode == 'global_to_single':
            used_split_idx = 0 if region_i!=generalization_region_i else orig_split_idx # can be 0 or 1
        else:
            raise NotImplementedError
        # used_split_idx = 0 if region_i!=generalization_region_i else 1 # orig_split_idx
        # print([a.shape for a in x_data[orig_split_idx]])
        joined_x = np.concatenate(x_data[orig_split_idx], axis=2)
        joined_y = np.concatenate(y_data[orig_split_idx])
        # print(joined_x.shape, joined_y.shape)
        processed_values[used_split_idx].append(joined_x) # .transpose([2, 0, 1]))
        # adding in region_i to track
        processed_targets_with_region_is[used_split_idx].append((joined_y, region_i))
# Now processed is same as above, list of 5 (per region) with train and test



# train_values = [vals[0] for i,vals in enumerate(processed_values) if i!=generalization_region_i]
# train_targets = [targets[0] for i,targets in enumerate(processed_targets) if i!=generalization_region_i]
# could do below in fancy list comp but this suffices
train_values = []
train_targets = []
for vals, (targets, region_i) in zip(processed_values[0], processed_targets_with_region_is[0]):
    # if region_i==generalization_region_i: continue
    train_values.append(vals)
    # print(vals.shape, targets.shape)
    train_targets.append(targets)
train_x = np.concatenate(train_values, axis=2).transpose(2, 0, 1)
train_x = train_x.reshape(train_x.shape[0], -1)
train_y = np.concatenate(train_targets)
print("Train data shapes", train_x.shape, train_y.shape)

test_values = []
test_targets = []
for vals, (targets, region_i) in zip(processed_values[1], processed_targets_with_region_is[1]):
    # if region_i!=generalization_region_i: continue
    test_values.append(vals)
    # print(vals.shape, targets.shape)
    test_targets.append(targets)
test_x = np.concatenate(test_values, axis=2).transpose(2, 0, 1)
test_x = test_x.reshape(test_x.shape[0], -1)
test_y = np.concatenate(test_targets)
subsample_freq = 1
test_x = test_x[::subsample_freq]
test_y = test_y[::subsample_freq]
print("Test data shapes", test_x.shape, test_y.shape)

"""
test_values = [vals[1] for i,vals in enumerate(processed_values) if i==generalization_region_i]
test_targets = [targets[1] for i,targets in enumerate(processed_targets) if i==generalization_region_i]
test_x = np.concatenate(test_values, axis=2).transpose(2, 0, 1)
test_x = test_x.reshape(test_x.shape[0], -1)
test_y = np.concatenate(test_targets)
"""

# test_x = processed_values[generalization_region_i].transpose([2, 0, 1])[::100].copy()
# test_y = processed_targets[generalization_region_i][::100].copy()
# train_x = all_x.transpose(2, 0, 1)
# train_y = all_y
# processed values is 5 (# regions) x n_points x 8 x 9


all_mean_reps = []
# num_points_to_average = int(sys.argv[3]) if len(sys.argv)>3 else None

class TorchMLP():
    def __init__(self, num_classes=len(interest_classes), num_hidden_layers=1, hidden_width=256, input_dim=72):
        self.num_classes = len(interest_classes)
        curr_dim = input_dim
        layers = []
        for _ in range(num_hidden_layers):
            layers.extend( [torch.nn.Linear(curr_dim, hidden_width), torch.nn.ReLU()])
            curr_dim = hidden_width
        layers.extend( [torch.nn.Linear(curr_dim, num_classes)] )
        mlp = torch.nn.Sequential(*layers)
        print("Cudaing MLP")
        self.mlp = mlp.cuda()
        self.orig_class_to_index = None

    def cast_targets(self, y):
        if self.orig_class_to_index is None:
            classes = sorted(list(set(y)))
            self.orig_class_to_index = {int(c):i for i,c in enumerate(classes)}
        new_targets = [self.orig_class_to_index[int(y_i)] for y_i in y]
        return new_targets

    def fit(self, train_x, train_y, num_epochs=1, bs=4096):
        x = torch.Tensor(train_x)
        n_train = x.shape[0]
        y = torch.LongTensor(self.cast_targets(train_y))
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
        opt = torch.optim.Adam(self.mlp.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        for epoch_i in range(num_epochs):
            num_seen = 0
            for bi, (bx, by) in enumerate(loader):
                bx = bx.cuda()
                by = by.cuda()
                num_seen += bx.shape[0]
                if not bi%20: print(f"{num_seen} / {n_train}")
                opt.zero_grad()
                preds = self.mlp(bx)
                loss = criterion(preds, by)
                loss.backward()
                opt.step()


class SoftDTWBarycenter():
    def __init__(self):
        pass

    def reshape_softdtw(self, s1, s2, n_time_points=8):
        feed1 = s1.reshape(n_time_points, -1)
        feed2 = s2.reshape(n_time_points, -1)
        return soft_dtw(feed1, feed2)


    def fit(self, train_x, train_y):
        centroids = []
        centroid_labels = []
        for class_i in interest_classes:
            print(f"Fitting class {class_i}")
            matching_idx = np.argwhere(train_y==class_i)
            matching_x = train_x[matching_idx].reshape(matching_idx.shape[0], 8, 9)
            centroid = softdtw_barycenter(matching_x)
            centroids.append(centroid)
            centroid_labels.append(class_i)
        self.clf = neighbors.KNeighborsClassifier(metric=self.reshape_softdtw)
        self.clf.fit(centroids.reshape(len(interest_classes), -1), centroid_labels)


    def score(self, test_x, test_y):
        print("Re-using soft dtw instead of hard, validate decision")
        return self.clf.score(test_x, test_y)

class MeanBarycenter():
    def __init__(self):
        pass

    def fit(self, train_x, train_y):
        centroids = []
        centroid_labels = []
        k = 1
        for class_i in interest_classes:
            matching_idx = np.argwhere(train_y==class_i).squeeze()
            # print(class_i, matching_idx.shape)
            centroid = train_x[matching_idx].mean(axis=0).squeeze()
            # print(centroid.shape)
            centroids.append(centroid)
            centroid_labels.append(class_i)
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
        print(k, centroid_labels, len(centroids))
        self.clf.fit(centroids, centroid_labels)
        print(self.clf.get_params())

    def score(self, test_x, test_y):
        return self.clf.score(test_x, test_y)


clean_drop_channels = [4, 5, 7, 8]
IR_channels = [7, 8]
# clf_strs=['mlp', 'logistic']
# clf_strs=['softdtw_centroid', 'euc_centroid_custom', 'euc_centroid']
clf_strs=['euc_centroid']
# clf_strs= ['euc_knn_5', 'euc_knn_3', 'euc_knn_1',
#             'dtw_knn_5', 'dtw_knn_3', 'dtw_knn_1',
#            'logistic']
#  data_prep_strs = ['', 'normalize', 'clean_drop', 'ir_drop', 'normalize+clean_drop', "normalize+ir_drop", "ndvi", "normalize+ndvi"]
data_prep_strs = ['']
for clf_str in clf_strs:
    for data_prep in data_prep_strs:
            print(clf_str, data_prep)
            print("Initializing classifier")
            if 'knn' in clf_str:
                n_neighbors = int(clf_str.split('knn_')[1])
                weights = 'distance'
                if 'euc_knn' in clf_str:
                    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                         weights=weights)
                if 'dtw_knn' in clf_str:
                    clf = neighbors.KNeighborsClassifier(metric=reshape_dtw,
                                                         n_neighbors=n_neighbors,
                                                          weights=weights)
            elif clf_str == 'softdtw_centroid':
                clf = SoftDTWBarycenter()
            elif clf_str == 'euc_centroid':
                clf = neighbors.NearestCentroid()
            elif clf_str == 'euc_centroid_custom':
                clf = MeanBarycenter()
            elif clf_str == 'logistic':
                clf = linear_model.LogisticRegression()
            elif clf_str == 'mlp':
                clf = neural_network.MLPClassifier()
                # clf = linear_model.LogisticRegression()
            elif clf_str == 'linear':
                clf = TorchMLP(num_hidden_layers=0)
            elif clf_str == 'TorchMLP':
                clf = TorchMLP()
            else:
                raise NotImplementedError
            # print("RUNNING ON 1/100 DATA")
            data_prep_list = data_prep.split('+')
            # samples x dates x channels
            if 'normalize' in data_prep_list:
                train_x = (train_x - train_x.mean(axis=(0,1))) / train_x.std(axis=(0,1))
                test_x = (test_x - test_x.mean(axis=(0,1))) / train_x.std(axis=(0,1))
            if 'clean_drop' in data_prep_list:
                train_x = np.delete(train_x, clean_drop_channels, axis=2)
                test_x = np.delete(test_x, clean_drop_channels, axis=2)
            if 'ir_drop' in data_prep_list:
                train_x = np.delete(train_x, IR_channels, axis=2)
                test_x = np.delete(test_x, IR_channels, axis=2)
            if 'ndvi' in data_prep_list:
                def ndvi(x):
                    nir = x[:, :, 4]
                    r = x[:, :, 3]
                    return (nir - r) / (nir + r)
                train_x = ndvi(train_x)
                test_x = ndvi(test_x)
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)
            clf.fit(train_x, train_y)
            gen_score = clf.score(test_x, test_y)
            # metrics.plot_confusion_matrix(clf, test_x, test_y)
            # plt.savefig(f"{generalization_region}_{clf_str}_{data_prep}.png")
            # pred_test_y = clf.predict(test_x)
            # confusion_matrix = metrics.confusion_matrix(test_y, pred_test_y, labels=interest_classes)
            # np.save(f"{generalization_region}_{clf_str}_{data_prep}.npy", confusion_matrix)
            print(clf_str, data_prep, gen_score)
            sys.stdout.flush()
