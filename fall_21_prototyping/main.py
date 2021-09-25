import rasterio
from scipy import optimize, spatial
from sklearn import neighbors, linear_model, metrics, cluster
from tslearn.metrics import dtw, soft_dtw
from tslearn.barycenters import dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter, euclidean_barycenter
import os
import numpy as np
import pickle
from collections import Counter
from interest_classes import interest_classes
import random
from pprint import pprint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import torch
from copy import copy
import argparse
from transformer import Transformer
from scipy import stats
from mcr_loss import MaximalCodingRateReduction
import autograd_hacks
from opacus import PrivacyEngine

print(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument("--generalization-region", type=str, default='')
parser.add_argument("--source-region", type=str, default='')
parser.add_argument("--save-basedir", type=str, default="/share/bharath/bw462/sat/knn_caching/")
parser.add_argument("--data-mode", type=str, default="generalization")
parser.add_argument("--clf-strs", type=str, nargs='+', default=['euc_centroid'])
parser.add_argument("--data-prep-strs", type=str, nargs='+', default=[''])
parser.add_argument("--train-subsample-freq", type=int, default=1)
parser.add_argument("--test-subsample-freq", type=int, default=1)
parser.add_argument("--num-ntk-nets", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--mlp-width", type=int, default=256)
parser.add_argument("--thresh", type=float, default=0.5, help="generic threshold")
args = parser.parse_args()

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
save_basedir = args.save_basedir

regions = os.listdir(basedir)
all_dirs = []
for region in regions:
    region_basedir = f"{basedir}/{region}"
    region_dirs = [f"{region_basedir}/{d}/2017/" for d in os.listdir(region_basedir)]
    all_dirs.append(region_dirs)

# dates are the same
start_month = 4
end_month = 7
dates = [d for d in os.listdir(all_dirs[0][0]) if os.path.isdir(f"{all_dirs[0][0]}/{d}") and int(d.split('-')[0]) in range(start_month, end_month+1)]
dates = sorted(dates)
print("Dates for data:", dates)

generalization_region = args.generalization_region # e.g. us_south
generalization_region_i = regions.index(generalization_region) if generalization_region in regions else np.inf
if not args.generalization_region: assert args.data_mode == 'global_to_global'

if args.source_region:
    assert args.data_mode == 'single_to_single'
    source_region = args.source_region
    source_region_i = regions.index(source_region)


values = []
targets = []
for region, dir_group in zip(regions, all_dirs):
    print(f"Loading {region}")
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
            # print(f"Loading from {save_dir}")
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
        sub_values[split_idx].append(x)
        sub_targets[split_idx].append(y)
    values.append(copy(sub_values))
    targets.append(copy(sub_targets))


# values and targets are still of length 5 but have train and test components to them
#
# data settings are "generalization", "direct" and "global_to_single"
# generalization = train on A-D, generalize to E (test split)
# direct train/test on splits of E
# global_to_single = fusion of two above train sets
processed_values = [[], []]
processed_targets_with_region_is = [[], []]
data_mode = args.data_mode
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
        elif data_mode == 'single_to_single':
            if orig_split_idx==0 and region_i==source_region_i:
                pass
            elif orig_split_idx==1 and region_i==generalization_region_i:
                pass
            else:
                continue
            used_split_idx = orig_split_idx
        elif data_mode == 'global_to_single':
            used_split_idx = 0 if region_i!=generalization_region_i else orig_split_idx # can be 0 or 1
        elif data_mode == 'global_to_global':
            used_split_idx = orig_split_idx
        else:
            raise NotImplementedError
        print(regions[region_i], orig_split_idx, len(x_data[orig_split_idx]), used_split_idx)
        # used_split_idx = 0 if region_i!=generalization_region_i else 1 # orig_split_idx
        joined_x = np.concatenate(x_data[orig_split_idx], axis=2)
        joined_y = np.concatenate(y_data[orig_split_idx])
        processed_values[used_split_idx].append(joined_x) # .transpose([2, 0, 1]))
        # adding in region_i to track
        processed_targets_with_region_is[used_split_idx].append((joined_y, region_i))
# Now processed is same as above, list of 5 (per region) with train and test



# could do below in fancy list comp but this suffices
train_values = []
train_targets = []
train_regions = []
region_class_hash_increment = 1e6
for vals, (targets, region_i) in zip(processed_values[0], processed_targets_with_region_is[0]):
    train_values.append(vals)
    # For now need to manipulate data so can only all per_region or all not
    is_per_region_arr = ['per_region' in s for s in args.clf_strs]
    if any(is_per_region_arr): assert all(is_per_region_arr)
    if is_per_region_arr[0]:
        targets += region_class_hash_increment * region_i
    train_targets.append(targets)
    region_name = regions[region_i]
    if region_name not in train_regions: train_regions.append(regions[region_i])
train_x = np.concatenate(train_values, axis=2).transpose(2, 0, 1)
# train_x = train_x.reshape(train_x.shape[0], -1)
train_y = np.concatenate(train_targets)
train_subsample_freq = args.train_subsample_freq
train_x = train_x[::train_subsample_freq]
train_y = train_y[::train_subsample_freq]
print("Train data shapes", train_x.shape, train_y.shape)

test_values = []
test_targets = []
for vals, (targets, region_i) in zip(processed_values[1], processed_targets_with_region_is[1]):
    test_values.append(vals)
    test_targets.append(targets)
test_x = np.concatenate(test_values, axis=2).transpose(2, 0, 1)
# test_x = test_x.reshape(test_x.shape[0], -1)
test_y = np.concatenate(test_targets)
test_subsample_freq = args.test_subsample_freq
test_x = test_x[::test_subsample_freq]
test_y = test_y[::test_subsample_freq]
print("Test data shapes", test_x.shape, test_y.shape)


all_mean_reps = []

class TorchNN():
    def __init__(self, num_classes=len(interest_classes), num_hidden_layers=1, hidden_width=256, wd=0, use_bn=True):
        input_dim = train_x.shape[1]
        
        self.orig_class_to_index = None
        print("TODO: Dynamically adapt dimension of clf layer")
        self.num_classes = len(interest_classes)

        self.hidden_width = hidden_width
        curr_dim = input_dim
        layers = []
        for _ in range(num_hidden_layers):
            layers.extend( [torch.nn.Linear(curr_dim, hidden_width), torch.nn.ReLU(), torch.nn.BatchNorm1d(hidden_width) if use_bn else torch.nn.Identity()])
            curr_dim = hidden_width
        layers.extend( [torch.nn.Linear(curr_dim, num_classes)] )
        mlp = torch.nn.Sequential(*layers)
        # print("Cudaing NN")
        self.mlp = mlp.cuda()
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=1e-2, weight_decay=wd)

    def cast_targets(self, y):
        if self.orig_class_to_index is None:
            self.seen_classes = set(list(y))
            self.orig_class_to_index = {int(c):i for i,c in enumerate(interest_classes)}
            # print(self.orig_class_to_index)
        else:
            new_classes = set(y) - self.seen_classes
            if len(new_classes):
                print(f"Previously unseen classes {new_classes}")
                num_unseen = 0
                for c in new_classes:
                    num_unseen += (y == c).sum()
                frac_unseen = num_unseen / y.shape[0]
                print(f"These account for {frac_unseen} of testing data")

        new_targets = [self.orig_class_to_index[int(y_i)] for y_i in y]
        return new_targets

    def fit(self, train_x, train_y, bs=4096, return_best_val_acc=False,
             silent=False, sample_weights=None):
        print_call = (lambda x: None) if silent else print
        self.mlp.train()
        x = torch.Tensor(train_x)
        n_train = x.shape[0]
        y = torch.LongTensor(self.cast_targets(train_y))
        if sample_weights is None:
            sample_weights = torch.ones(x.shape[0]).cuda()
        dataset = torch.utils.data.TensorDataset(x, y, sample_weights)
        num_val_points = n_train // 10
        num_train_points = n_train - num_val_points
        train_set, val_set = torch.utils.data.random_split(dataset, [num_train_points, num_val_points])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', patience=5)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        epoch_i = 0
        lr_steps = 0
        best_val_loss = np.inf
        best_val_acc = - np.inf
        min_loss_epoch = None
        while lr_steps <= 2:
            num_seen = 0
            num_correct = 0
            loss_sum = 0
            for bi, (bx, by, batch_weights) in enumerate(train_loader):
                bx = bx.cuda()
                by = by.cuda()
                curr_bs = bx.shape[0]
                num_seen += curr_bs
                # if not bi%500: print_call(f"{num_seen} / {n_train}")
                self.opt.zero_grad()
                preds = self.mlp(bx)
                batch_weights = curr_bs * batch_weights / batch_weights.sum()
                loss = (criterion(preds, by) * batch_weights).mean()
                loss.backward()
                self.opt.step()
                num_correct += (preds.argmax(dim=1) == by).sum().item()
                loss_sum += curr_bs * loss.item()
            ave_loss = loss_sum / num_seen
            if not epoch_i % 10:
                print_call(f"Train Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
                print_call(f"Train Loss @ Epoch {epoch_i}: {ave_loss}")

            with torch.no_grad():
                num_seen = 0
                num_correct = 0
                loss_sum = 0
                for bi, (bx, by, batch_weights) in enumerate(val_loader):
                    bx = bx.cuda()
                    by = by.cuda()
                    curr_bs = bx.shape[0]
                    num_seen += curr_bs
                    # if not bi%500: print_call(f"{num_seen} / {n_train}")
                    preds = self.mlp(bx)
                    # get total weight equal to curr_bs
                    batch_weights = curr_bs * batch_weights / batch_weights.sum()
                    loss = (criterion(preds, by) * batch_weights).mean()
                    num_correct += (preds.argmax(dim=1) == by).sum().item()
                    loss_sum += curr_bs * loss.item()
                ave_loss = loss_sum / num_seen
                val_acc = num_correct / num_seen
                if val_acc > best_val_acc: best_val_acc = val_acc
                if not epoch_i % 20:
                    print_call(f"Val Acc @ Epoch {epoch_i}: {val_acc}")
                    print_call(f"Val Loss @ Epoch {epoch_i}: {ave_loss}")
                    print_call(f"Best Val Loss was {best_val_loss} @ Epoch {min_loss_epoch}")

                curr_lr = self.opt.state_dict()['param_groups'][0]['lr']
                scheduler.step(ave_loss)
                if curr_lr != self.opt.state_dict()['param_groups'][0]['lr']:
                    print_call("Stepped LR")
                    lr_steps += 1
                if ave_loss < best_val_loss:
                    best_sd = self.mlp.state_dict()
                    best_val_loss = ave_loss
                    min_loss_epoch = epoch_i

            epoch_i += 1
        self.mlp.load_state_dict(best_sd)
        if return_best_val_acc: return best_val_acc


    def score(self, test_x, test_y, bs=4096):
        with torch.no_grad():
            self.mlp.eval()
            x = torch.Tensor(test_x)
            n_train = x.shape[0]
            y = torch.LongTensor(self.cast_targets(test_y))
            dataset = torch.utils.data.TensorDataset(x, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
            criterion = torch.nn.CrossEntropyLoss()
            num_seen = 0
            num_correct = 0
            for bi, (bx, by) in enumerate(loader):
                bx = bx.cuda()
                by = by.cuda()
                num_seen += bx.shape[0]
                # if not bi%20: print(f"{num_seen} / {n_train}")
                preds = self.mlp(bx)
                loss = criterion(preds, by)
                num_correct += (preds.argmax(dim=1) == by).sum().item()
            return num_correct / num_seen

    def predict(self, test_x, bs=4096, return_vec=False):
        with torch.no_grad():
            self.mlp.eval()
            x = torch.Tensor(test_x)
            n_train = x.shape[0]
            dataset = torch.utils.data.TensorDataset(x)
            loader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                                 batch_size=bs)
            preds_list = []
            for bi, (bx,) in enumerate(loader):
                bx = bx.cuda()
                output = self.mlp(bx)
                if not return_vec: output = output.argmax(dim=1)
                preds_list.append(output)
            return torch.cat(preds_list)



class TransformerNN(TorchNN):
    def __init__(self, num_classes=len(interest_classes), in_channels=9, t_len=8,
                 n_conv=2, **kwargs):
        super().__init__(num_classes=num_classes)
        mlp = Transformer(num_classes=num_classes, in_channels=in_channels, n_conv=n_conv, **kwargs)

        # print(f"CUDA there?: {torch.cuda.is_available()}")
        # print("Re-cudaing Transformer NN if needed")
        self.mlp = mlp.cuda()
        self.opt = torch.optim.Adam(self.mlp.parameters())

class TargetClassesTransformerNN():
    def __init__(self, *args, **kwargs):
        self.transformer_constructor = lambda : TransformerNN(*args, **kwargs)

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def score(self, test_x, test_y):
        target_classes = list(set(list(test_y)))
        interest_idx = np.argwhere(np.isin(self.train_y, target_classes)).squeeze()
        train_y = self.train_y[interest_idx]
        train_x = self.train_x[interest_idx]
        clf = self.transformer_constructor()
        clf.fit(train_x, train_y)
        return clf.score(test_x, test_y)


class RetrainTransformerNN():
    def __init__(self, *args, **kwargs):
        self.transformer_constructor = lambda : TransformerNN(*args, **kwargs)

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def score(self, test_x, test_y):
        initial_transformer = self.transformer_constructor()
        train_y_preds = initial_transformer.predict(self.train_x)
        correct_mask = (train_y_preds.cpu() == torch.Tensor(initial_transformer.cast_targets(self.train_y)))
        print(correct_mask.sum().item() / correct_mask.shape[0])
        initial_transformer.fit(self.train_x, self.train_y)
        print("Consider dataset split consequences here")
        train_y_preds = initial_transformer.predict(self.train_x)
        correct_mask = (train_y_preds.cpu() == torch.Tensor(initial_transformer.cast_targets(self.train_y)))
        print(correct_mask.sum().item() / correct_mask.shape[0])

        new_transformer = self.transformer_constructor()
        retrain_idx = np.argwhere(correct_mask.numpy())
        print(retrain_idx.shape[0], correct_mask.sum())
        new_transformer.fit(self.train_x[retrain_idx].squeeze(),
                            self.train_y[retrain_idx].squeeze())
        return new_transformer.score(test_x, test_y)


class TransformerEnsemble():
    def __init__(self, method='average', *args, **kwargs):
        self.method = method
        self.transformer_constructor = lambda : TransformerNN(*args, **kwargs)

    def fit(self, train_x, train_y):
        # This gives region indices for each transformer
        region_assignments = train_y // region_class_hash_increment
        self.region_class_factors = sorted(set(list(region_assignments)))
        self.num_regions = len(self.region_class_factors)
        self.transformers = [self.transformer_constructor() for _ in range(self.num_regions)]
        for net in self.transformers: net.mlp.train()
        print(self.region_class_factors)
        per_region_masks = [region_assignments==r for r in self.region_class_factors]
        per_region_x = [train_x[mask] for mask in per_region_masks]
        per_region_y = [train_y[mask] for mask in per_region_masks]
        for region_i, (region_x, region_y, net) in enumerate(zip(per_region_x, per_region_y, self.transformers)):
            print(region_i, region_x.shape, region_y.shape, net)
            net.fit(region_x, region_y % region_class_hash_increment)

    def score(self, test_x, test_y, bs=4096):
        with torch.no_grad():
            for net in self.transformers: net.mlp.eval()
            x = torch.Tensor(test_x)
            n_train = x.shape[0]
            print("Ignore any missing class region")
            y = torch.LongTensor(self.transformers[0].cast_targets(test_y % region_class_hash_increment))
            dataset = torch.utils.data.TensorDataset(x, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
            num_seen = 0
            num_correct = 0
            for bi, (bx, by) in enumerate(loader):
                bx = bx.cuda()
                by = by.cuda()
                num_seen += bx.shape[0]
                # if not bi%20: print(f"{num_seen} / {n_train}")
                all_probs = torch.stack([net.mlp(bx).softmax(dim=1) for net in self.transformers])
                if self.method == 'average':
                    preds = torch.mean(all_probs, dim=0)
                elif self.method == 'highest_confidence':
                    # n_point x classes x regions
                    transposed_probs = all_probs.permute(1, 2, 0)
                    # Below isn't softmaxxed but still n_points x n_class
                    preds = transposed_probs.max(dim=2).values
                    # should work below
                else:
                    raise NotImplementedError
                num_correct += (preds.argmax(dim=1) == by).sum().item()
            return num_correct / num_seen

class HDivergence():
    def __init__(self, *args, **kwargs):
        self.transformer_constructor = lambda : TransformerNN(num_classes=2, *args, **kwargs)

    def fit(self, train_x, train_y):
        # This gives region indices for each transformer
        region_assignments = train_y // region_class_hash_increment
        self.region_class_factors = sorted(set(list(region_assignments)))
        self.num_regions = len(self.region_class_factors)
        per_region_masks = [region_assignments==r for r in self.region_class_factors]
        self.per_region_x = [train_x[mask] for mask in per_region_masks]
        self.per_region_y = [train_y[mask] for mask in per_region_masks]


    def score(self, test_x, test_y):
        h_div_accs = []
        for region_train_x, region_train_y in zip(self.per_region_x, self.per_region_y):
            transformer = self.transformer_constructor()
            x = np.concatenate([region_train_x, test_x])
            # class 1 is corn and 5 is soybeans both of which are interest classes
            # so be having 1/5 as targets they get casted properly
            y = np.concatenate([np.ones(region_train_x.shape[0]),
                                5 * np.ones(test_x.shape[0])])
            h_div_acc = transformer.fit(x, y, return_best_val_acc = True)
            h_div_accs.append(h_div_acc)
        selected_region_i = np.argmin(h_div_accs)
        print("Use score to figure out which was selected")
        print(train_regions[selected_region_i])
        # return self.clfs[selected_region_i].score(test_x, test_y)

class GeneralizingHDivergence():
    # Idea is to train 3 vs 1 region and use that to evaluate whether to include
    # class from new region
    def __init__(self, thresh=0.5, group_by_class=False, in_channels=9, *args, **kwargs):
        self.thresh = thresh
        self.transformer_constructor = lambda n: TransformerNN(num_classes=n,
                                                              in_channels=in_channels,
                                                              *args, **kwargs)
        self.group_by_class = group_by_class

    def fit(self, train_x, train_y):
        # This gives region indices for each transformer
        region_assignments = train_y // region_class_hash_increment
        self.region_class_factors = sorted(set(list(region_assignments)))
        self.num_regions = len(self.region_class_factors)
        per_region_masks = [region_assignments==r for r in self.region_class_factors]
        self.per_region_x = [train_x[mask] for mask in per_region_masks]
        self.per_region_y = [train_y[mask] for mask in per_region_masks]


    def score(self, test_x, test_y):
        # doing this per-point right now instead of per-class
        classified_as_test_x = []
        classified_as_test_y = []
        for train_region_i in range(len(self.per_region_x)):
            print(train_region_i)
            left_out_x = self.per_region_x[train_region_i]
            left_out_y = self.per_region_y[train_region_i]
            included_x = self.per_region_x[:train_region_i] + self.per_region_x[train_region_i+1:]
            used_train_x = np.concatenate(included_x)
            transformer = self.transformer_constructor(2)
            print(train_regions[train_region_i])
            print(used_train_x.shape, test_x.shape)
            x = np.concatenate([used_train_x, test_x])
            # class 1 is corn and 5 is soybeans both of which are interest classes
            # so be having 1/5 as targets they get casted properly
            y = np.concatenate([np.ones(used_train_x.shape[0]),
                                5 * np.ones(test_x.shape[0])])
            transformer.fit(x, y, silent=True)
            preds = transformer.predict(left_out_x, return_vec=True)
            print(preds.shape)
            probs = preds.softmax(dim=1)[:, 1].cpu()
            print(probs, probs.min(), probs.max(), probs.mean())
            if self.group_by_class:
                # want to key on left_out_y
                for class_i in set(list(left_out_y)):
                    class_idx = (left_out_y == class_i).squeeze()
                    class_probs = probs[class_idx]
                    average_conf = class_probs.mean()
                    print(class_i, average_conf)
                    if average_conf >= self.thresh:
                        classified_as_test_x.append(left_out_x[class_idx])
                        classified_as_test_y.append(left_out_y[class_idx])
                        print(class_idx.shape, left_out_x[class_idx].shape, left_out_y[class_idx].shape)
            else:
                take_idx = np.argwhere((probs >= self.thresh).numpy()).squeeze()
                classified_as_test_x.append(left_out_x[take_idx])
                classified_as_test_y.append(left_out_y[take_idx])
        new_train_x = np.concatenate(classified_as_test_x)
        new_train_y = np.concatenate(classified_as_test_y) %region_class_hash_increment
        print(new_train_x.shape, new_train_y.shape)
        new_transformer = self.transformer_constructor(len(interest_classes))
        new_transformer.fit(new_train_x, new_train_y, silent=True)
        return new_transformer.score(test_x, test_y)

class NTK():
    # Idea is calculate gradient of MCR with respect to pseudoclusters and compare that
    # to MCR with respect to true labels on source domain
    def __init__(self, num_ntk_nets, dimension=32, group_by_class_and_region=True):
        self.num_ntk_nets = num_ntk_nets
        self.dimension = dimension
        print("Using MLP for now b/c transformer layers are incompatible")
        self.transformer_constructor = lambda n, use_bn: TorchNN(num_classes=n, use_bn=use_bn)
        # self.transformer_constructor = lambda n: TransformerNN(num_classes=n)
        self.group_by_class_and_region = group_by_class_and_region

    def fit(self, train_x, train_y):
        # not doing per-region stuff here
        self.train_x = train_x
        self.train_y = train_y
        if self.group_by_class_and_region:
            region_assignments = train_y // region_class_hash_increment
            self.region_class_factors = sorted(set(list(region_assignments)))
            self.num_regions = len(self.region_class_factors)
            per_region_masks = [region_assignments==r for r in self.region_class_factors]
            self.per_region_x = [train_x[mask] for mask in per_region_masks]
            self.per_region_y = [train_y[mask] for mask in per_region_masks]


    def get_ave_grad_vec(self, net):
        grad_list = []
        for param in net.parameters():
            grad = param.grad
            grad_list.append(param.grad.flatten())
            if hasattr(param, 'grad_sample'):
                print(param.grad_sample.shape)
                # Note that this is off by factor of 1024
                print(param.grad_sample.sum(), grad.sum())
                # assert param.grad_sample.sum() == grad.sum()
        return torch.cat(grad_list)

    def get_indiv_grad_vec(self, net):
        grad_list = []
        for param in net.parameters():
            grad = param.grad_sample
            grad_list.append(grad.flatten(start_dim=1))
        return torch.cat(grad_list, dim=1)
    

    def score(self, test_x, test_y, bs=1024):
        criterion = MaximalCodingRateReduction()
        weightings = []
        for net_i in range(self.num_ntk_nets):
            print(f"Net {net_i+1} / {self.num_ntk_nets}")
            net = self.transformer_constructor(self.dimension, False) # True)
            print("Need to be in train mode b/c of privacy engine")
            net.mlp.train() # eval()
            print("Currently doing supervised transductive training with test_y, unacceptable long term")
            dataset = torch.utils.data.TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
            loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                                 batch_size=bs)
            # autograd hacks only is built for conv1d/linear so don't know how to go about transformer encoder
            # just doing incremental batch size for now, stupid slow but get things rolling
            # autograd_hacks.add_hooks(net.mlp)
            for bi, (bx,by) in enumerate(loader):
                bx = bx.cuda()
                # by = by.cuda()
                output = net.mlp(bx)
                loss, _, _ = criterion(output, by)
                loss.backward()
                # autograd_hacks.compute_grad1(net.mlp)
            gen_region_grad = self.get_ave_grad_vec(net.mlp)
            print(gen_region_grad.shape, gen_region_grad.sum())
            optim = torch.optim.Adam(net.mlp.parameters())
            privacy_engine = PrivacyEngine(net.mlp, max_grad_norm=np.inf, batch_size=bs,
                                            sample_size=self.train_x.shape[0],
                                            noise_multiplier=0)
            privacy_engine.attach(optim)
            optim.zero_grad()
            if self.group_by_class_and_region:
                region_and_class_i_to_agreements = {}
                # want to key on left_out_y
                for region_i, (region_x, region_y) in enumerate(zip(self.per_region_x, self.per_region_y)):
                    for class_i in set(list(region_y)):
                        region_class_tuple = (region_i, class_i)
                        if region_class_tuple not in region_and_class_i_to_agreements:
                            region_and_class_i_to_agreements[region_class_tuple] = []

                        class_idx = (region_y == class_i).squeeze()
                        class_x = region_x[class_idx]
                        class_y = region_y[class_idx]
                        dataset = torch.utils.data.TensorDataset(torch.Tensor(class_x), torch.Tensor(class_y))
                        loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=bs)
                        for bi, (bx,by) in enumerate(loader):
                            bx = bx.cuda()
                            # by = by.cuda()
                            output = net.mlp(bx)
                            loss, _, _ = criterion(output, by)
                            loss.backward()
                            grad = self.get_grad_vec(net.mlp)
                            optim.zero_grad()
                        similarity = (gen_region_grad * grad).sum()
                        region_and_class_i_to_agreements[region_class_tuple].append(similarity)
                        print("Ending here b/c switching to soft system idea")
                        raise NotImplementedError
            else:
                dataset = torch.utils.data.TensorDataset(torch.arange(self.train_x.shape[0]),
                                                         torch.Tensor(self.train_x),
                                                         torch.Tensor(self.train_y))
                loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                                     batch_size=bs)
                grad_list = []
                idx_list = []
                for bi, (data_idx, bx,by) in enumerate(loader):
                    if not bi%10: print(f"{bi} / {len(loader)}")
                    bx = bx.cuda()
                    output = net.mlp(bx)
                    loss, _, _ = criterion(output, by)
                    loss.backward()
                    grad_list.append(self.get_indiv_grad_vec(net.mlp))
                    idx_list.append(data_idx)
                    optim.zero_grad()
                indiv_grads = torch.cat(grad_list)
                grads_order = torch.cat(idx_list)
                # need to reverse intuition, doing indiv_grads[grads_order] would
                # mess up b/c if idx 0 was the 100th point looked at then this
                # would make the 100th entry of indiv_grads whatever the first one was
                sort_idx = torch.argsort(grads_order)
                indiv_grads = indiv_grads[sort_idx]
            # gen_region_Grad is M
            # indiv_grads is n x M
            # Want to find vector v (n) s.t. v.sigmoid()* indiv_grads/n = gen_region_grad
            # (changed ablove slightly, now have gen grad normalized and want to find
            # positive coefficients
            gen_region_grad = gen_region_grad / gen_region_grad.norm()
            # Analysis: look at distribution of dot products of gen_region with indiv
            #
            # normed_indiv = indiv_grads / (indiv_grads.norm(dim=1, keepdim=True) + 1e-8)
            # sims = torch.matmul(normed_indiv, gen_region_grad)
            # print(sims.min(), sims.max(), sims.mean(), sims.std())
            # example prints of this
            # tensor(-0.5717, device='cuda:0') tensor(0.5315, device='cuda:0') tensor(-0.0019, device='cuda:0') tensor(0.1834, device='cuda:0')
            # tensor(-0.7319, device='cuda:0') tensor(0.5516, device='cuda:0') tensor(-0.0118, device='cuda:0') tensor(0.1968, device='cuda:0')
            # REALLY want to do version of this in reverse where we find points that by
            # training incorrectly on can get non-trivial performance
            if True:
                normed_indiv = indiv_grads / (indiv_grads.norm(dim=1, keepdim=True) + 1e-8)
                weighting = torch.matmul(normed_indiv, gen_region_grad).relu()
            else:
                # switched from this b/c optimal solution is just whichever
                # vector is closest. Could do unscaled, but then have weird relationship
                # between # of available points and recon acc, also slower
                pre_act_weights = torch.nn.Parameter(torch.ones(indiv_grads.shape[0]).cuda())
                # optim = torch.optim.Adam([pre_act_weights])
                optim = torch.optim.SGD([pre_act_weights], lr=10)
                # below optimization sis slowing or more points
                for linear_i in range(10000):
                    optim.zero_grad()
                    post_act_weights = pre_act_weights.relu()
                    reconstruction_unnormed = torch.matmul(indiv_grads.t(), post_act_weights)
                    reconstruction = reconstruction_unnormed / reconstruction_unnormed.norm()
                    loss = -1 * torch.dot(reconstruction, gen_region_grad)
                    loss.backward()
                    optim.step()
                    if not linear_i%1000: print(linear_i, loss)
                print(post_act_weights.min(), post_act_weights.max(), post_act_weights.mean(), post_act_weights.std(), post_act_weights.sum())
                weighting = post_act_weights.data.detach()
            weightings.append(weighting)
        if weightings:
            weightings = torch.stack(weightings)
            print(weightings.shape)
            weightings = weightings.mean(dim=0).detach()
            print(weightings.min(), weightings.max(), weightings.mean(), weightings.std())
        else:
            weightings = None
        # Then retrain
        new_transformer = self.transformer_constructor(len(interest_classes), True)
        new_transformer.fit(self.train_x, self.train_y, silent=True, sample_weights=weightings)
        return new_transformer.score(test_x, test_y)


class HDivergenceSorting():
    def __init__(self, num_training_trials=10, *args, **kwargs):
        self.num_training_trials =  num_training_trials
        self.transformer_constructor = lambda : TransformerNN(num_classes=2, *args, **kwargs)

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y


    def score(self, test_x, test_y):
        transformer = self.transformer_constructor()
        x = np.concatenate([self.train_x, test_x])
        # class 1 is corn and 5 is soybeans both of which are interest classes
        # so be having 1/5 as targets they get casted properly
        # These correspond with targets of 0 and 1
        # So basically relative weight of being in val
        y = np.concatenate([np.ones(self.train_x.shape[0]),
                            5 * np.ones(test_x.shape[0])])
        transformer.fit(x, y)
        # below is the non-softmaxed weights of which to predict
        # slicing should be undeeded now that have only 2 classes
        preds = transformer.predict(self.train_x, return_vec=True)[:, :2]
        # making sure that network learned to ignore other bits (comment out above slicing)
        # print(preds.softmax(dim=1)[:, :2].sum(dim=1).min(), preds.softmax(dim=1)[:, :2].sum(dim=1).mean())
        probs = preds.softmax(dim=1)[:, 1].cpu()
        # example stats from below print statement
        # torch.Size([17782]) tensor(7.4238e-05, device='cuda:0') tensor(0.9902, device='cuda:0') tensor(0.0041, device='cuda:0')

        # print(probs.shape, probs.min(), probs.max(), probs.mean())
        q_tensor = torch.arange(99, 0, -1) / 100
        q_values = torch.quantile(probs, q_tensor)
        print(*zip(q_tensor, q_values))
        all_accs = []
        for q, v in zip(q_tensor, q_values):
            print(q, v)
            mask = (probs >= v)
            retrain_idx = np.argwhere(mask.numpy())
            q_accs = []
            for trial_i in range(self.num_training_trials):
                new_transformer = self.transformer_constructor()
                new_transformer.fit(self.train_x[retrain_idx].squeeze(),
                                    self.train_y[retrain_idx].squeeze(),
                                    silent=True)
                q_acc = new_transformer.score(test_x, test_y)
                q_accs.append(q_acc)
                print(q_acc)
            q_ave_acc = np.average(q_accs)
            print(q_ave_acc)
            all_accs.append(q_ave_acc)
        return all_accs


class KMeansEvaluation():
    def __init__(self, n_clusters, *args, **kwargs):
        self.kmeans_constructor = lambda : cluster.KMeans(n_clusters=n_clusters, *args, **kwargs)
        self.n_clusters = n_clusters

    def fit(self, train_x, train_y):
        pass

    def score(self, test_x, test_y):
        test_kmeans = self.kmeans_constructor()
        assignments = test_kmeans.fit_predict(test_x, test_y)
        print("Homogeneity score", metrics.homogeneity_score(test_y, assignments))
        print(assignments.min(), assignments.max())
        assignment_to_class = {}
        for assignment_i in range(self.n_clusters):
            assigned_data_mask = (assignments==assignment_i)
            true_labels = test_y[assigned_data_mask]
            cluster_label = stats.mode(true_labels)[0][0]
            assignment_to_class[assignment_i] = cluster_label
        class_preds = np.array([assignment_to_class[y] for y in assignments])
        correct = (class_preds == test_y)
        print(assignment_to_class)
        print(Counter(assignment_to_class.values()))
        print(correct.shape)
        return correct.sum() / correct.shape[0]


class KMeansMatching():
    def __init__(self, *args, **kwargs):
        self.kmeans_constructor = lambda : cluster.KMeans(*args, **kwargs)

    def fit(self, train_x, train_y):
        # This gives region indices for each transformer
        region_assignments = train_y // region_class_hash_increment
        self.region_class_factors = sorted(set(list(region_assignments)))
        self.num_regions = len(self.region_class_factors)
        self.clfs = [self.kmeans_constructor() for _ in range(self.num_regions)]
        per_region_masks = [region_assignments==r for r in self.region_class_factors]
        per_region_x = [train_x[mask] for mask in per_region_masks]
        per_region_y = [train_y[mask] for mask in per_region_masks]
        self.centers_list = []
        for region_i, (region_x, region_y, clf) in enumerate(zip(per_region_x, per_region_y, self.clfs)):
            print(region_i, region_x.shape, region_y.shape, clf)
            clf.fit(region_x, region_y % region_class_hash_increment)
            self.centers_list.append(clf.cluster_centers_)


    def score(self, test_x, test_y):
        test_kmeans = self.kmeans_constructor()
        test_kmeans.fit(test_x, test_y)
        test_centers = test_kmeans.cluster_centers_
        matching_costs = []
        for train_centers in self.centers_list:
            cost_matrix = spatial.distance.cdist(train_centers, test_centers)
            sol_row_ind, sol_col_ind = optimize.linear_sum_assignment(cost_matrix)
            cost = cost_matrix[sol_row_ind, sol_col_ind].sum()
            matching_costs.append(cost)
        selected_region_i = np.argmin(matching_costs)
        print("Use score to figure out which was selected")
        print(train_regions[selected_region_i])
        # return self.clfs[selected_region_i].score(test_x, test_y)



class TransductiveStartupNN(TorchNN):

    def __init__(self, *args, **kwargs):
        super(TransductiveStartupNN, self).__init__(*args, **kwargs)


    def transductive_fit(self, train_x, train_y, test_x, trans_y, bs=4096):
        # startup doesn't use any weighting between two losses
        # Reinit MLP head
        self.mlp.train()
        print("Reinitializing head and transductive fitting")
        self.mlp[-1].load_state_dict(torch.nn.Linear(self.hidden_width, self.num_classes).state_dict())

        # make loaders
        x = torch.Tensor(train_x)
        n_train = x.shape[0]
        y = torch.LongTensor(self.cast_targets(train_y))
        dataset = torch.utils.data.TensorDataset(x, y)
        num_val_points = n_train // 10
        num_train_points = n_train - num_val_points
        train_set, val_set = torch.utils.data.random_split(dataset, [num_train_points, num_val_points])

        trans_x = torch.Tensor(test_x)
        n_trans = trans_x.shape[0]
        print(len(trans_y))
        print(trans_x.shape, trans_y.shape)
        trans_dataset = torch.utils.data.TensorDataset(trans_x, trans_y)
        num_trans_val_points = n_trans // 10
        num_trans_train_points = n_trans - num_trans_val_points
        trans_train_set, trans_val_set = torch.utils.data.random_split(trans_dataset, [num_trans_train_points, num_trans_val_points])


        train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
        trans_train_loader = torch.utils.data.DataLoader(trans_train_set, batch_size=bs, shuffle=True)

        # shuffling val so can match to length of trans data
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True)
        trans_val_loader = torch.utils.data.DataLoader(trans_val_set, batch_size=bs, shuffle=True)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', patience=5)
        criterion = torch.nn.CrossEntropyLoss()
        trans_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        epoch_i = 0
        lr_steps = 0
        best_val_loss = np.inf
        min_loss_epoch = None
        while lr_steps <= 2:
            num_seen = 0
            # num_correct = 0
            loss_sum = 0
            for bi, ((bx, by), (btx, bty)) in enumerate(zip(train_loader, trans_train_loader)):
                bx = bx.cuda()
                by = by.cuda()
                btx = btx.cuda()
                bty = bty.cuda()
                curr_bs = bx.shape[0]
                num_seen += curr_bs
                # if not bi%500: print(f"{num_seen} / {min(num_train_points, num_trans_train_points)}")
                self.opt.zero_grad()
                train_preds = self.mlp(bx)
                train_loss = criterion(train_preds, by)
                trans_preds = self.mlp(btx).softmax(dim=1)
                trans_loss = trans_criterion(trans_preds, bty)
                loss = train_loss + trans_loss
                loss.backward()
                self.opt.step()
                # num_correct += (preds.argmax(dim=1) == by).sum().item()
                loss_sum += curr_bs * loss.item()
            ave_loss = loss_sum / num_seen
            # print(f"Train Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
            if not epoch_i % 10:
                print(f"Transductive Train Loss @ Epoch {epoch_i}: {ave_loss}")

            with torch.no_grad():
                num_seen = 0
                # num_correct = 0
                loss_sum = 0
                for bi, ((bx, by), (btx, bty)) in enumerate(zip(val_loader, trans_val_loader)):
                    bx = bx.cuda()
                    by = by.cuda()
                    btx = btx.cuda()
                    bty = bty.cuda()
                    curr_bs = bx.shape[0]
                    num_seen += curr_bs
                    # if not bi%500: print(f"{num_seen} / {min(num_val_points, num_trans_val_points)}")
                    test_preds = self.mlp(bx)
                    test_loss = criterion(test_preds, by)
                    trans_preds = self.mlp(btx).softmax(dim=1)
                    trans_loss = trans_criterion(trans_preds, bty)
                    loss = test_loss + trans_loss
                    # num_correct += (preds.argmax(dim=1) == by).sum().item()
                    loss_sum += curr_bs * loss.item()
                ave_loss = loss_sum / num_seen
                # print(f"Val Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
                if not epoch_i % 10:
                    print(f"Transductive Val Loss @ Epoch {epoch_i}: {ave_loss}")
                    print(f"Best Val Loss was {best_val_loss} @ Epoch {min_loss_epoch}")
                curr_lr = self.opt.state_dict()['param_groups'][0]['lr']
                scheduler.step(ave_loss)
                if curr_lr != self.opt.state_dict()['param_groups'][0]['lr']:
                    print("Stepped LR")
                    lr_steps += 1
                if ave_loss < best_val_loss:
                    best_sd = self.mlp.state_dict()
                    min_loss_epoch = epoch_i
                    best_val_loss = ave_loss

            epoch_i += 1
        self.mlp.load_state_dict(best_sd)

    def transductive_score(self, train_x, train_y, test_x, test_y, bs=4096):
        print(f"Initial score {self.score(test_x, test_y)}")
        with torch.no_grad():
            self.mlp.eval()
            x = torch.Tensor(test_x)
            n_train = x.shape[0]
            dataset = torch.utils.data.TensorDataset(x)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
            soft_targets = []
            for (bx,) in loader:
                soft_targets.append(self.mlp(bx.cuda()).softmax(dim=1))
            trans_y = torch.cat(soft_targets)
        print("Consider making fit finite number of epochs instead of early stopping")
        self.transductive_fit(train_x, train_y, test_x, trans_y, bs=bs)
        # transductive fit along with original train data
        #
        return self.score(test_x, test_y)

class TransductiveHardNN(TorchNN):

    def __init__(self, confidence_threshold=0, *args, **kwargs):
        super(TransductiveHardNN, self).__init__(*args, **kwargs)
        self.thresh = confidence_threshold

    def transductive_fit(self, train_x, train_y, test_x, bs=4096):
        self.mlp.train()
        x = torch.Tensor(train_x)
        n_train = x.shape[0]
        y = torch.LongTensor(self.cast_targets(train_y))
        dataset = torch.utils.data.TensorDataset(x, y)
        num_val_points = n_train // 10
        num_train_points = n_train - num_val_points
        train_set, val_set = torch.utils.data.random_split(dataset, [num_train_points, num_val_points])

        trans_x = torch.Tensor(test_x)
        print(trans_x.shape)
        trans_dataset = torch.utils.data.TensorDataset(trans_x)
        n_trans = trans_x.shape[0]
        num_trans_val_points = n_trans // 10
        num_trans_train_points = n_trans - num_trans_val_points
        print(len(trans_dataset))
        print(n_trans, num_trans_val_points, num_trans_train_points)
        trans_train_set, trans_val_set = torch.utils.data.random_split(trans_dataset, [num_trans_train_points, num_trans_val_points])


        train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
        trans_train_loader = torch.utils.data.DataLoader(trans_train_set, batch_size=bs, shuffle=True)

        # shuffling val so can match to length of trans data
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True)
        trans_val_loader = torch.utils.data.DataLoader(trans_val_set, batch_size=bs, shuffle=True)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', patience=5)
        criterion = torch.nn.CrossEntropyLoss()
        epoch_i = 0
        lr_steps = 0
        best_val_loss = np.inf
        min_loss_epoch = None
        while lr_steps <= 2:
            num_seen = 0
            # num_correct = 0
            loss_sum = 0
            for bi, ((bx, by), (btx,)) in enumerate(zip(train_loader, trans_train_loader)):
                bx = bx.cuda()
                by = by.cuda()
                btx = btx.cuda()
                curr_bs = bx.shape[0]
                num_seen += curr_bs
                # if not bi%500: print(f"{num_seen} / {min(num_train_points, num_trans_train_points)}")
                self.opt.zero_grad()
                train_preds = self.mlp(bx)
                train_loss = criterion(train_preds, by)
                trans_preds = self.mlp(btx)
                trans_probs = trans_preds.softmax(dim=1)
                confident_mask = trans_probs.max(dim=1)[0] > self.thresh
                confident_preds = trans_preds[confident_mask]
                trans_loss = criterion(confident_preds, confident_preds.argmax(dim=1))
                loss = train_loss + trans_loss
                loss.backward()
                self.opt.step()
                # num_correct += (preds.argmax(dim=1) == by).sum().item()
                loss_sum += curr_bs * loss.item()
            ave_loss = loss_sum / num_seen
            # print(f"Train Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
            if not epoch_i % 10:
                print(f"Transductive Train Loss @ Epoch {epoch_i}: {ave_loss}")

            with torch.no_grad():
                num_seen = 0
                # num_correct = 0
                loss_sum = 0
                for bi, ((bx, by), (btx,)) in enumerate(zip(val_loader, trans_val_loader)):
                    bx = bx.cuda()
                    by = by.cuda()
                    btx = btx.cuda()
                    curr_bs = bx.shape[0]
                    num_seen += curr_bs
                    # if not bi%500: print(f"{num_seen} / {min(num_val_points, num_trans_val_points)}")
                    train_preds = self.mlp(bx)
                    train_loss = criterion(train_preds, by)
                    trans_preds = self.mlp(btx)
                    trans_probs = trans_preds.softmax(dim=1)
                    confident_mask = trans_probs.max(dim=1)[0] > self.thresh
                    confident_preds = trans_preds[confident_mask]
                    trans_loss = criterion(confident_preds, confident_preds.argmax(dim=1))
                    loss = train_loss + trans_loss
                    # num_correct += (preds.argmax(dim=1) == by).sum().item()
                    loss_sum += curr_bs * loss.item()
                ave_loss = loss_sum / num_seen
                # print(f"Val Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
                if not epoch_i % 10:
                    print(f"Transductive Val Loss @ Epoch {epoch_i}: {ave_loss}")
                    print(f"Best Val Loss was {best_val_loss} @ Epoch {min_loss_epoch}")

                curr_lr = self.opt.state_dict()['param_groups'][0]['lr']
                scheduler.step(ave_loss)
                if curr_lr != self.opt.state_dict()['param_groups'][0]['lr']:
                    print("Stepped LR")
                    lr_steps += 1
                if ave_loss < best_val_loss:
                    best_sd = self.mlp.state_dict()
                    best_val_loss = ave_loss
                    min_loss_epoch = epoch_i

            epoch_i += 1
        self.mlp.load_state_dict(best_sd)


    def transductive_score(self, train_x, train_y, test_x, test_y, bs=4096):
        print(f"Initial score {self.score(test_x, test_y)}")
        self.transductive_fit(train_x, train_y, test_x, bs=4096)
        return self.score(test_x, test_y)



class DTWBarycenter():
    def __init__(self, barycenter_method):
        self.barycenter_method = barycenter_method
        if barycenter_method == 'soft':
            self.barycenter_call = lambda x: softdtw_barycenter(x, max_iter=5)
        elif barycenter_method == 'SG':
            self.barycenter_call = lambda x: dtw_barycenter_averaging_subgradient(x, max_iter=5)
        elif barycenter_method == 'hybrid':
            self.barycenter_call = euclidean_barycenter
        else:
            raise NotImplementedError

    def reshape_dtw(self, s1, s2, n_time_points=8):
        feed1 = s1.reshape(n_time_points, -1)
        feed2 = s2.reshape(n_time_points, -1)
        return soft_dtw(feed1, feed2) if self.barycenter_method=='soft' else dtw(feed1, feed2)


    def fit(self, train_x, train_y):
        centroids = []
        centroid_labels = []
        self.seen_classes = set()
        for class_i in interest_classes:
            print(f"Fitting class {class_i}")
            matching_idx = np.argwhere(train_y==class_i)
            if not len(matching_idx): continue
            self.seen_classes.add(class_i)
            matching_x = train_x[matching_idx].reshape(matching_idx.shape[0], 8, -1)
            centroid = self.barycenter_call(matching_x)
            centroids.append(centroid)
            centroid_labels.append(class_i)
        self.clf = neighbors.KNeighborsClassifier(metric=self.reshape_dtw)
        self.clf.fit(np.array(centroids).reshape(len(self.seen_classes), -1), centroid_labels)


    def score(self, test_x, test_y):
        new_classes = set(y) - self.seen_classes
        if len(new_classes):
            print(f"Previously unseen classes {new_classes}")
            num_unseen = 0
            for c in new_classes:
                num_unseen += (y == c).sum()
            frac_unseen = num_unseen / y.shape[0]
            print(f"These account for {frac_unseen} of testing data")
        return self.clf.score(test_x, test_y)


    def transductive_fit(self, train_x, train_y, test_x, test_y, num_iter=100):
        for iter_i in range(num_iter):
            print(f"Iteration {iter_i}")
            pseudo_y = self.predict(test_x)
            concat_x = np.concatenate([train_x, test_x])
            concat_y = np.concatenate([train_y, pseudo_y])
            clf.fit(concat_x, concat_y)
            print(self.score(test_x, test_y))


    def transductive_score(self, train_x, train_y, test_x, test_y):
        self.transductive_fit(train_x, train_y, test_x, test_y)
        return self.score(test_x, test_y)



class TransductiveEucCentroid(neighbors.NearestCentroid):

    def __init__(self, *args, **kwargs):
        super(TransductiveEucCentroid, self).__init__(*args, **kwargs)


    def transductive_fit(self, train_x, train_y, test_x, test_y, num_iter=100):
        for iter_i in range(num_iter):
            print(f"Iteration {iter_i}")
            pseudo_y = self.predict(test_x)
            concat_x = np.concatenate([train_x, test_x])
            concat_y = np.concatenate([train_y, pseudo_y])
            clf.fit(concat_x, concat_y)
            print(self.score(test_x, test_y))

    def transductive_score(self, train_x, train_y, test_x, test_y):
        self.transductive_fit(train_x, train_y, test_x, test_y)
        return self.score(test_x, test_y)


class PerRegionNearestCentroid():

    def __init__(self, n_neighbors=1, weights='distance'):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def fit(self, train_x, train_y):
        per_region_centroid_clf = neighbors.NearestCentroid()
        per_region_centroid_clf.fit(train_x, train_y)
        centroids = per_region_centroid_clf.centroids_
        classes = per_region_centroid_clf.classes_
        true_classes = classes % region_class_hash_increment
        self.clf.fit(centroids, true_classes)

    def score(self, test_x, test_y):
        return self.clf.score(test_x, test_y)


clean_drop_channels = [4, 5, 7, 8]
IR_channels = [7, 8]
clf_strs = args.clf_strs
data_prep_strs = args.data_prep_strs
for clf_str in clf_strs:
    for data_prep in data_prep_strs:
            print(clf_str, data_prep)
            data_prep_list = data_prep.split('+')
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
            elif 'softdtw_centroid' in clf_str:# might have _transducitve at front
                clf =  DTWBarycenter('soft')
            elif 'SGdtw_centroid' in clf_str:# might have _transducitve at front
                clf =  DTWBarycenter('SG')
            elif clf_str == 'euc_centroid':
                clf = neighbors.NearestCentroid()
            elif clf_str == 'per_region_euc_centroid':
                clf = PerRegionNearestCentroid()
            elif clf_str == 'hybrid_barycenter':
                clf = DTWBarycenter('hybrid')
            elif clf_str == 'transductive_euc_centroid':
                clf = TransductiveEucCentroid()
            elif clf_str == 'logistic':
                clf = linear_model.LogisticRegression()
            elif clf_str == 'mlp':
                clf = TorchNN()
            elif clf_str == 'mlp_no_bn':
                clf = TorchNN(use_bn=False)
            elif clf_str == 'mlp_wd':
                clf = TorchNN(wd=1e-2)
            elif clf_str == 'transductive_startup_mlp':
                clf = TransductiveStartupNN()
            elif clf_str == 'transductive_hard_mlp':
                clf = TransductiveHardNN()
            elif clf_str == 'linear':
                clf = TorchNN(num_hidden_layers=0)
            elif clf_str == 'transformer':
                clf = TransformerNN()
            elif clf_str == 'retrain_transformer':
                clf = RetrainTransformerNN()
            elif clf_str == 'transformer_target_classes_only':
                clf = TargetClassesTransformerNN()
            elif clf_str == 'kmeans_eval':
                clf = KMeansEvaluation(n_clusters=len(interest_classes))
            elif clf_str == 'per_region_transformer_average_ensemble':
                clf = TransformerEnsemble()
            elif clf_str == 'per_region_transformer_confidence_ensemble':
                clf = TransformerEnsemble(method='highest_confidence')
            elif clf_str == 'per_region_kmeans_matching':
                clf = KMeansMatching(n_clusters=50)
            elif clf_str == 'per_region_h_div':
                clf = HDivergence()
            elif clf_str == 'h_div_select':
                clf = HDivergenceSorting()
            elif clf_str == 'per_region_gen_h_div':
                clf = GeneralizingHDivergence(thresh=args.thresh, group_by_class=False,
                                            in_channels=7 if 'ir_drop' in data_prep_list else 9)
            elif clf_str == 'per_region_classwise_gen_h_div':
                clf = GeneralizingHDivergence(thresh=args.thresh, group_by_class=True,
                                            in_channels=7 if 'ir_drop' in data_prep_list else 9)
            elif clf_str == 'ntk':
                clf = NTK(args.num_ntk_nets, group_by_class_and_region=False)
            elif clf_str == 'per_region_ntk':
                clf = NTK(args.num_ntk_nets, group_by_class_and_region=True)
            else:
                raise NotImplementedError
            clf.fit(train_x, train_y)
            gen_score = clf.transductive_score(train_x, train_y, test_x, test_y) if 'transductive' in clf_str else clf.score(test_x, test_y, bs=args.batch_size)
            # metrics.plot_confusion_matrix(clf, test_x, test_y)
            # plt.savefig(f"{generalization_region}_{clf_str}_{data_prep}.png")
            # pred_test_y = clf.predict(test_x)
            # confusion_matrix = metrics.confusion_matrix(test_y, pred_test_y, labels=interest_classes)
            # np.save(f"{generalization_region}_{clf_str}_{data_prep}.npy", confusion_matrix)
            print(clf_str, data_prep, gen_score)
            sys.stdout.flush()
