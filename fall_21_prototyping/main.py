import rasterio
from sklearn import neighbors, linear_model, metrics, neural_network 
from tslearn.metrics import dtw, soft_dtw
from tslearn.barycenters import dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter
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

print(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument("--generalization-region", type=str, required=True)
parser.add_argument("--save-basedir", type=str, default="/share/bharath/bw462/sat/knn_caching/")
parser.add_argument("--data-mode", type=str, default="generalization")
parser.add_argument("--clf-strs", type=str, nargs='+', default=['euc_centroid'])
parser.add_argument("--data-prep-strs", type=str, nargs='+', default=[''])
parser.add_argument("--train-subsample-freq", type=int, default=1)
parser.add_argument("--test-subsample-freq", type=int, default=1)
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
generalization_region_i = regions.index(generalization_region)

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
        elif data_mode == 'global_to_single':
            used_split_idx = 0 if region_i!=generalization_region_i else orig_split_idx # can be 0 or 1
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
for vals, (targets, region_i) in zip(processed_values[0], processed_targets_with_region_is[0]):
    train_values.append(vals)
    train_targets.append(targets)
train_x = np.concatenate(train_values, axis=2).transpose(2, 0, 1)
train_x = train_x.reshape(train_x.shape[0], -1)
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
test_x = test_x.reshape(test_x.shape[0], -1)
test_y = np.concatenate(test_targets)
test_subsample_freq = args.test_subsample_freq
test_x = test_x[::test_subsample_freq]
test_y = test_y[::test_subsample_freq]
print("Test data shapes", test_x.shape, test_y.shape)


all_mean_reps = []

class TorchNN():
    def __init__(self, num_classes=len(interest_classes), num_hidden_layers=1, hidden_width=256, input_dim=72):
        self.num_classes = len(interest_classes)
        self.hidden_width = hidden_width
        curr_dim = input_dim
        layers = []
        for _ in range(num_hidden_layers):
            layers.extend( [torch.nn.Linear(curr_dim, hidden_width), torch.nn.ReLU(), torch.nn.BatchNorm1d(hidden_width)])
            curr_dim = hidden_width
        layers.extend( [torch.nn.Linear(curr_dim, num_classes)] )
        mlp = torch.nn.Sequential(*layers)
        print("Cudaing NN")
        self.mlp = mlp.cuda()
        self.orig_class_to_index = None
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=1e-2)

    def cast_targets(self, y):
        if self.orig_class_to_index is None:
            self.seen_classes = set(y)
            self.orig_class_to_index = {int(c):i for i,c in enumerate(interest_classes)}
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

    def fit(self, train_x, train_y, bs=4096):
        x = torch.Tensor(train_x)
        n_train = x.shape[0]
        y = torch.LongTensor(self.cast_targets(train_y))
        dataset = torch.utils.data.TensorDataset(x, y)
        num_val_points = n_train // 10
        num_train_points = n_train - num_val_points
        train_set, val_set = torch.utils.data.random_split(dataset, [num_train_points, num_val_points])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', patience=5)
        criterion = torch.nn.CrossEntropyLoss()
        epoch_i = 0
        lr_steps = 0
        best_val_loss = np.inf
        min_loss_epoch = None
        while lr_steps <= 2:
            num_seen = 0
            num_correct = 0
            loss_sum = 0
            for bi, (bx, by) in enumerate(train_loader):
                bx = bx.cuda()
                by = by.cuda()
                curr_bs = bx.shape[0]
                num_seen += curr_bs
                if not bi%500: print(f"{num_seen} / {n_train}")
                self.opt.zero_grad()
                preds = self.mlp(bx)
                loss = criterion(preds, by)
                loss.backward()
                self.opt.step()
                num_correct += (preds.argmax(dim=1) == by).sum().item()
                loss_sum += curr_bs * loss.item()
            ave_loss = loss_sum / num_seen
            print(f"Train Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
            print(f"Train Loss @ Epoch {epoch_i}: {ave_loss}")

            with torch.no_grad():
                num_seen = 0
                num_correct = 0
                loss_sum = 0
                for bi, (bx, by) in enumerate(val_loader):
                    bx = bx.cuda()
                    by = by.cuda()
                    curr_bs = bx.shape[0]
                    num_seen += curr_bs
                    if not bi%500: print(f"{num_seen} / {n_train}")
                    preds = self.mlp(bx)
                    loss = criterion(preds, by)
                    num_correct += (preds.argmax(dim=1) == by).sum().item()
                    loss_sum += curr_bs * loss.item()
                ave_loss = loss_sum / num_seen
                print(f"Val Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
                print(f"Val Loss @ Epoch {epoch_i}: {ave_loss}")
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
            print()
        self.mlp.load_state_dict(best_sd)


    def score(self, test_x, test_y, bs=4096):
        with torch.no_grad():
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
                if not bi%20: print(f"{num_seen} / {n_train}")
                preds = self.mlp(bx)
                loss = criterion(preds, by)
                num_correct += (preds.argmax(dim=1) == by).sum().item()
            return num_correct / num_seen


class TransductiveStartupNN(TorchNN):

    def __init__(self, *args, **kwargs):
        super(TransductiveStartupNN, self).__init__(*args, **kwargs)



    def transductive_fit(self, train_x, train_y, test_x, trans_y, bs=4096):
        # startup doesn't use any weighting between two losses
        # Reinit MLP head
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
                if not bi%500: print(f"{num_seen} / {min(num_train_points, num_trans_train_points)}")
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
                    if not bi%500: print(f"{num_seen} / {min(num_val_points, num_trans_val_points)}")
                    test_preds = self.mlp(bx)
                    test_loss = criterion(test_preds, by)
                    trans_preds = self.mlp(btx).softmax(dim=1)
                    trans_loss = trans_criterion(trans_preds, bty)
                    loss = test_loss + trans_loss
                    # num_correct += (preds.argmax(dim=1) == by).sum().item()
                    loss_sum += curr_bs * loss.item()
                ave_loss = loss_sum / num_seen
                # print(f"Val Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
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
            print()
        self.mlp.load_state_dict(best_sd)

    def transductive_score(self, train_x, train_y, test_x, test_y, bs=4096):
        print(f"Initial score {self.score(test_x, test_y)}")
        with torch.no_grad():
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
                if not bi%500: print(f"{num_seen} / {min(num_train_points, num_trans_train_points)}")
                self.opt.zero_grad()
                train_preds = self.mlp(bx)
                train_loss = criterion(train_preds, by)
                trans_preds = self.mlp(btx)
                trans_probs = trans_preds.softmax(dim=1)
                confident_mask = trans_probs.max(dim=1)[0] > self.thresh
                confident_preds = trans_preds[confident_mask]
                confident_inputs = btx[confident_mask]
                trans_loss = criterion(confident_preds, confident_preds.argmax(dim=1))
                loss = train_loss + trans_loss
                loss.backward()
                self.opt.step()
                # num_correct += (preds.argmax(dim=1) == by).sum().item()
                loss_sum += curr_bs * loss.item()
            ave_loss = loss_sum / num_seen
            # print(f"Train Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
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
                    if not bi%500: print(f"{num_seen} / {min(num_val_points, num_trans_val_points)}")
                    train_preds = self.mlp(bx)
                    train_loss = criterion(train_preds, by)
                    trans_preds = self.mlp(btx).softmax()
                    trans_loss = trans_criterion(trans_preds, btx)
                    loss = train_loss + trans_loss
                    loss.backward()
                    # num_correct += (preds.argmax(dim=1) == by).sum().item()
                    loss_sum += curr_bs * loss.item()
                ave_loss = loss_sum / num_seen
                # print(f"Val Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
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
            print()
        self.mlp.load_state_dict(best_sd)


    def transductive_score(self, train_x, train_y, test_x, test_y, bs=4096):
        print(f"Initial score {self.score(test_x, test_y)}")
        self.transductive_fit(train_x, train_y, test_x, bs=4096)
        return self.score(test_x, test_y)



class DTWBarycenter():
    def __init__(self, barycenter_method):
        self.barycenter_method = barycenter_method
        if barycenter_method == 'soft':
            self.barycenter_call = softdtw_barycenter
        elif barycenter_method == 'SG':
            self.barycenter_call = dtw_barycenter_averaging_subgradient
        else:
            raise NotImplementedError

    def reshape_softdtw(self, s1, s2, n_time_points=8):
        feed1 = s1.reshape(n_time_points, -1)
        feed2 = s2.reshape(n_time_points, -1)
        return soft_dtw(feed1, feed2)


    def fit(self, train_x, train_y):
        centroids = []
        centroid_labels = []
        self.seen_classes = set()
        for class_i in interest_classes:
            print(f"Fitting class {class_i}")
            matching_idx = np.argwhere(train_y==class_i)
            if not len(matching_idx): continue
            self.seen_classes.add(class_i)
            matching_x = train_x[matching_idx].reshape(matching_idx.shape[0], 8, 9)
            centroid = self.barycenter_call(matching_x, max_iter=5)
            centroids.append(centroid)
            centroid_labels.append(class_i)
        self.clf = neighbors.KNeighborsClassifier(metric=self.reshape_softdtw)
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
        print("Re-using soft dtw instead of hard, validate decision")
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


clean_drop_channels = [4, 5, 7, 8]
IR_channels = [7, 8]
clf_strs = args.clf_strs
data_prep_strs = args.data_prep_strs
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
            elif 'softdtw_centroid' in clf_str:# might have _transducitve at front
                clf =  DTWBarycenter('soft')
            elif 'SGdtw_centroid' in clf_str:# might have _transducitve at front
                clf =  DTWBarycenter('SG')
            elif clf_str == 'euc_centroid':
                clf = neighbors.NearestCentroid()
            elif clf_str == 'transductive_euc_centroid':
                clf = TransductiveEucCentroid()
            elif clf_str == 'euc_centroid_custom':
                clf = MeanBarycenter()
            elif clf_str == 'logistic':
                clf = linear_model.LogisticRegression()
            elif clf_str == 'mlp':
                clf = TorchNN()
            elif clf_str == 'transductive_startup_mlp':
                clf = TransductiveStartupNN()
            elif clf_str == 'transductive_hard_mlp':
                clf = TransductiveHardNN()
            elif clf_str == 'linear':
                clf = TorchNN(num_hidden_layers=0)
            else:
                raise NotImplementedError
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
            clf.fit(train_x, train_y)
            gen_score = clf.transductive_score(train_x, train_y, test_x, test_y) if 'transductive' in clf_str else clf.score(test_x, test_y)
            # metrics.plot_confusion_matrix(clf, test_x, test_y)
            # plt.savefig(f"{generalization_region}_{clf_str}_{data_prep}.png")
            # pred_test_y = clf.predict(test_x)
            # confusion_matrix = metrics.confusion_matrix(test_y, pred_test_y, labels=interest_classes)
            # np.save(f"{generalization_region}_{clf_str}_{data_prep}.npy", confusion_matrix)
            print(clf_str, data_prep, gen_score)
            sys.stdout.flush()
