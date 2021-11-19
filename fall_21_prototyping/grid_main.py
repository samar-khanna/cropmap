import rasterio
from scipy import optimize, spatial
from sklearn import neighbors, linear_model, metrics, cluster
from tslearn.metrics import dtw, soft_dtw
from tslearn.barycenters import dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter, \
    euclidean_barycenter
import os
import numpy as np
import pickle
from collections import Counter
from interest_classes import interest_classes
from util import get_accuracies
import random
from pprint import pprint
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
from copy import copy
import argparse
from transformer import Transformer
from scipy import stats
from mcr_loss import MaximalCodingRateReduction

print(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument("--generalization-region", type=str, default='')
parser.add_argument("--source-region", type=str, default='')
parser.add_argument("--save-basedir", type=str, default="/share/bharath/sak296/grid_1609/experiments")
parser.add_argument("--data-mode", type=str, default="generalization")
parser.add_argument("--clf-strs", type=str, nargs='+', default=['euc_centroid'])
parser.add_argument("--data-prep-strs", type=str, nargs='+', default=[''])
parser.add_argument("--train-subsample-freq", type=int, default=1)
parser.add_argument("--test-subsample-freq", type=int, default=1)
parser.add_argument("--num-ntk-nets", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--mlp-width", type=int, default=256)
parser.add_argument("--dimension", type=int, default=32)
parser.add_argument("--thresh", type=float, default=0.5, help="generic threshold")
parser.add_argument("--weight", type=float, default=0.0, help="generic weight")
parser.add_argument("--sample-weight", action='store_true', help="Reweight loss on source domain inv freq")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay value")
parser.add_argument("--greedy-similarity", action='store_true', help='NTK sim vs linear combo')
parser.add_argument("--norm-indivs", action='store_true', help='Normalize individual grads')
parser.add_argument("--checkpoint", type=str, default=None, help='Use checkpoint for scoring only')
args = parser.parse_args()

# IDK
region_class_hash_increment = 1e6

basedir = "/share/bharath/sak296/grid_1609/"
save_basedir = args.save_basedir
exp_save_dir = f"{'_'.join(args.clf_strs)}_{'_'.join(args.data_prep_strs)}" + \
                f"_te-{args.generalization_region}_tr-{args.source_region if args.source_region else 'all'}" + \
                (f"_w{args.weight}" if args.weight != 0.0 else '')
exp_save_dir = os.path.join(save_basedir, exp_save_dir)
os.makedirs(exp_save_dir, exist_ok=True)
prev_checkpoints = [int(n) for n in os.listdir(exp_save_dir) if n.isdigit()]
exp_save_dir = os.path.join(exp_save_dir, '0' if len(prev_checkpoints) == 0
                            else str(max(prev_checkpoints)+1))
os.makedirs(exp_save_dir, exist_ok=True)

regions = [f for f in os.listdir(basedir) if f.startswith('usa')]

generalization_region = args.generalization_region  # e.g. usa_g1
generalization_region_i = regions.index(generalization_region)
assert generalization_region in regions, "Need to specify gen region"

source_region = None
if args.source_region:
    # assert args.data_mode == 'single_to_single'
    source_region = args.source_region
    source_region_i = regions.index(source_region)


def load_data_from_pickle(path_dir):
    with open(os.path.join(path_dir, 'values.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(path_dir, 'labels.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(path_dir, 'coords.pkl'), 'rb') as f:
        coords = pickle.load(f)
    with open(os.path.join(path_dir, 'climate.pkl'), 'rb') as f:
        climate = pickle.load(f)
    return x, y, coords, climate


def load_and_append(region_dir, *buffers):
    load_out = load_data_from_pickle(region_dir)

    assert len(load_out) == len(buffers), "Provide same #args as pickle load items"
    for buffer, array in zip(buffers, load_out):
        buffer.append(array)


x_train, y_train, coords_train, climate_train = [], [], [], []
x_test, y_test, coords_test, climate_test = [], [], [], []
for i, region in enumerate(regions):
    region_dir = os.path.join(basedir, region)
    if i == generalization_region_i:
        load_and_append(region_dir, x_test, y_test, coords_test, climate_test)
        # x, y, coords, climate = load_data_from_pickle(os.path.join(basedir, region))
        # x_test.append(x)
        # y_test.append(y)
        # coords_test.append(coords)
    elif source_region is not None:
        # Specific source region in x_train
        if i == source_region_i:
            load_and_append(region_dir, x_train, y_train, coords_train, climate_train)
    else:
        # All source regions in x_train
        load_and_append(region_dir, x_train, y_train, coords_train, climate_train)

x_train = np.concatenate(x_train, axis=0)  # (N, c, t)
y_train = np.concatenate(y_train, axis=0)  # (N,)
coords_train = np.concatenate(coords_train, axis=0)  # (N, 2)  fmt: (lon, lat)
climate_train = np.concatenate(climate_train, axis=0)  # (N, 19)

x_test = np.concatenate(x_test, axis=0)  # (Nte, c, t)
y_test = np.concatenate(y_test, axis=0)  # (Nte,)
coords_test = np.concatenate(coords_test, axis=0)  # (Nte, 2)  fmt: (lon, lat)
climate_test = np.concatenate(climate_test, axis=0)  # (Nte, 19)

interest_train_mask = np.isin(y_train, interest_classes)
x_train = x_train[interest_train_mask]
y_train = y_train[interest_train_mask]
coords_train = coords_train[interest_train_mask]
climate_train = climate_train[interest_train_mask]

interest_test_mask = np.isin(y_test, interest_classes)
x_test = x_test[interest_test_mask]
y_test = y_test[interest_test_mask]
coords_test = coords_test[interest_test_mask]
climate_test = climate_test[interest_test_mask]

# TODO: Clear idx (filter away clouds)
# NOTE: Below clears away only FULLY clouded pixels (as we want)
cloud_train_mask = np.any(x_train.reshape(x_train.shape[0], -1) > 0, axis=-1)  # (N,)
x_train = x_train[cloud_train_mask]  # (N, c, t)
y_train = y_train[cloud_train_mask]
coords_train = coords_train[cloud_train_mask]
climate_train = climate_train[cloud_train_mask]

cloud_test_mask = np.any(x_test.reshape(x_test.shape[0], -1) > 0, axis=-1)  # (Nte,)
x_test = x_test[cloud_test_mask]
y_test = y_test[cloud_test_mask]
coords_test = coords_test[cloud_test_mask]
climate_test = climate_test[cloud_test_mask]

train_x = x_train.transpose(0, 2, 1)  # (N, t, c)
train_y = y_train

test_x = x_test.transpose(0, 2, 1)  # (Nte, t, c)
test_y = y_test

_, t, c = train_x.shape
print("Train data shapes", train_x.shape, train_y.shape)
print("Test data shapes", test_x.shape, test_y.shape)


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
            layers.extend([torch.nn.Linear(curr_dim, hidden_width), torch.nn.ReLU(),
                           torch.nn.BatchNorm1d(hidden_width) if use_bn else torch.nn.Identity()])
            curr_dim = hidden_width
        layers.extend([torch.nn.Linear(curr_dim, num_classes)])
        mlp = torch.nn.Sequential(*layers)
        # print("Cudaing NN")
        self.wd = wd
        self.mlp = mlp.cuda()
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=1e-3, weight_decay=wd)

    def cast_targets(self, y):
        if self.orig_class_to_index is None:
            self.seen_classes = set(list(y))
            self.orig_class_to_index = {int(c): i for i, c in enumerate(interest_classes)}
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
        sample_weights = torch.as_tensor(sample_weights).cuda() \
            if sample_weights is not None else torch.ones(x.shape[0]).cuda()

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

                    # TODO: Some logic to not always save?
                    torch.save(best_sd, os.path.join(exp_save_dir, 'checkpoint.bin'))

            epoch_i += 1
        self.mlp.load_state_dict(best_sd)
        if return_best_val_acc: return best_val_acc

    def score(self, test_x, test_y, bs=1024):
        with torch.no_grad():
            self.mlp.eval()
            x = torch.Tensor(test_x)
            n_train = x.shape[0]
            y = torch.LongTensor(self.cast_targets(test_y))
            dataset = torch.utils.data.TensorDataset(x, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
            criterion = torch.nn.CrossEntropyLoss()

            class_seen = Counter()
            class_correct = Counter()
            for bi, (bx, by) in enumerate(loader):
                bx = bx.cuda()
                by = by.cuda()
                # if not bi%20: print(f"{num_seen} / {n_train}")
                preds = self.mlp(bx)
                loss = criterion(preds, by)

                preds = preds.argmax(dim=1)

                batch_seen = Counter(by.cpu().numpy())
                batch_correct = {}
                for c in batch_seen:
                    class_mask = by == c
                    batch_correct[c] = (preds[class_mask] == by[class_mask]).sum().item()

                class_seen.update(batch_seen)
                class_correct.update(batch_correct)

            return get_accuracies(class_seen, class_correct)

    def predict(self, test_x, bs=1024, return_vec=False):
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
                 n_conv=2, wd=0, **kwargs):
        super().__init__(num_classes=num_classes, wd=wd)
        mlp = Transformer(num_classes=num_classes, in_channels=in_channels, n_conv=n_conv, **kwargs)

        self.mlp = mlp.cuda()
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=1e-3, weight_decay=wd)


class TransformerCorrelation(TransformerNN):
    def __init__(self, weight, reg_c, keep_reg, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.reg_c = reg_c
        self.keep_reg = keep_reg
        self.in_c = self.mlp.in_c if keep_reg else self.mlp.in_c + reg_c
        self.feat_lambda = 1e-3
        self.reg_weight = nn.Linear(self.mlp.dim_feature, 1).cuda()
        self.opt = torch.optim.Adam(list(self.mlp.parameters()) + list(self.reg_weight.parameters()),
                                    lr=1e-3, weight_decay=kwargs['wd'])

    def fit(self, train_x, train_y, bs=4096, return_best_val_acc=False,
            silent=False, sample_weights=None):
        print_call = (lambda x: None) if silent else print
        self.mlp.train()
        x = torch.Tensor(train_x)
        n_train = x.shape[0]
        y = torch.LongTensor(self.cast_targets(train_y))
        sample_weights = torch.as_tensor(sample_weights).cuda() \
            if sample_weights is not None else torch.ones(x.shape[0]).cuda()

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
        report_interval = 20
        while lr_steps <= 2:
            num_seen = 0
            num_correct = 0
            loss_sum = 0
            self.mlp.train()
            for bi, (bx, by, batch_weights) in enumerate(train_loader):
                bx = bx.cuda()
                N, num_channels = bx.shape
                orig_bx = bx.clone()
                bx = bx.view(N, -1, self.in_c)  # (N, t, in_c + extra_c)
                reg_t = bx[:, :, -self.reg_c:].mean(dim=1)  # all repeated anyways
                centered_reg_t = reg_t - reg_t.mean(0, keepdim=True)  #

                if self.keep_reg:
                    bx = bx.reshape(N, -1)  # (N, t*in_c*reg_c)
                else:
                    bx = bx[:, :, :-self.reg_c].reshape(N, -1)
                    assert abs(orig_bx.view(N, -1, self.in_c)[:, :, :-self.reg_c] -
                               bx.view(N, -1, self.mlp.in_c)).sum() < 1e-8

                by = by.cuda()
                curr_bs = bx.shape[0]
                num_seen += curr_bs

                self.opt.zero_grad()
                preds, feat = self.mlp(bx, return_final_feature=True)
                feat_m = (feat.T @ feat) + self.feat_lambda * torch.eye(feat.shape[-1]).cuda()  # (cxn @ nxc) + cxc = cxc
                # inv_feat_m = torch.cholesky_inverse(torch.cholesky(feat_m))  # (cxc)
                inv_feat_m = torch.linalg.inv(feat_m)
                coeffs = inv_feat_m @ feat.T @ centered_reg_t  # cxc @ cxn @ nxclim = cxclim
                # coeffs = feat.pinverse() @ centered_reg_t
                recon = feat @ coeffs  # nxc X cxclim = nxclim
                res = (centered_reg_t - recon).pow(2)  # n x clim
                # corr_loss = self.weight * res.mean() / (centered_reg_t.pow(2).mean())

                corr_weight = self.reg_weight(feat)
                corr_weight = self.weight * corr_weight
                corr_loss = (corr_weight * res).mean() / centered_reg_t.pow(2).mean()

                # res = torch.linalg.lstsq(feat, coords).residuals.mean()
                batch_weights = curr_bs * batch_weights / batch_weights.sum()
                clf_loss = (criterion(preds, by) * batch_weights).mean()
                # print(clf_loss, res, self.weight)
                # print(clf_loss, corr_loss)
                loss = clf_loss + corr_loss
                loss.backward()
                self.opt.step()
                num_correct += (preds.argmax(dim=1) == by).sum().item()
                loss_sum += curr_bs * loss.item()
            ave_loss = loss_sum / num_seen
            if not epoch_i % report_interval:
                print_call(f"Train Acc @ Epoch {epoch_i}: {num_correct / num_seen}")
                print_call(f"Train Loss @ Epoch {epoch_i}: {ave_loss}")

            with torch.no_grad():
                num_seen = 0
                num_correct = 0
                loss_sum = 0
                self.mlp.eval()
                for bi, (bx, by, batch_weights) in enumerate(val_loader):
                    bx = bx.cuda()
                    N, num_channels = bx.shape
                    orig_bx = bx.clone()
                    bx = bx.view(N, -1, self.in_c)  # (N, t, in_c + extra_c)
                    reg_t = bx[:, :, -self.reg_c:].mean(dim=1)  # all repeated anyways
                    centered_reg_t = reg_t - reg_t.mean(0, keepdim=True)  #

                    if self.keep_reg:
                        bx = bx.reshape(N, -1)  # (N, t*in_c*reg_c)
                    else:
                        bx = bx[:, :, :-self.reg_c].reshape(N, -1)
                        assert abs(orig_bx.view(N, -1, self.in_c)[:, :, :-self.reg_c] -
                                   bx.view(N, -1, self.mlp.in_c)).sum() < 1e-8

                    by = by.cuda()
                    curr_bs = bx.shape[0]
                    num_seen += curr_bs
                    # if not bi%500: print_call(f"{num_seen} / {n_train}")
                    preds, feat = self.mlp(bx, return_final_feature=True)
                    feat_m = (feat.T @ feat) + self.feat_lambda * torch.eye(feat.shape[-1]).cuda()  # (cxn @ nxc) + cxc = cxc
                    # inv_feat_m = torch.cholesky_inverse(torch.cholesky(feat_m))  # (cxc)
                    inv_feat_m = torch.linalg.inv(feat_m)
                    coeffs = inv_feat_m @ feat.T @ centered_reg_t  # cxc @ cxn @ nxclim = cxclim
                    # coeffs = feat.pinverse() @ centered_reg_t
                    recon = feat @ coeffs  # nxc X cxclim = nxclim
                    res = (centered_reg_t - recon).pow(2)
                    # corr_loss = self.weight * res.mean() / (centered_reg_t.pow(2).mean())

                    corr_weight = self.reg_weight(feat)
                    corr_weight = self.weight * corr_weight
                    corr_loss = (corr_weight * res).mean() / centered_reg_t.pow(2).mean()

                    # get total weight equal to curr_bs
                    batch_weights = curr_bs * batch_weights / batch_weights.sum()
                    clf_loss = (criterion(preds, by) * batch_weights).mean()
                    # print(clf_loss, res, self.weight)
                    loss = clf_loss + corr_loss
                    num_correct += (preds.argmax(dim=1) == by).sum().item()
                    loss_sum += curr_bs * loss.item()
                ave_loss = loss_sum / num_seen
                val_acc = num_correct / num_seen
                if val_acc > best_val_acc: best_val_acc = val_acc
                if not epoch_i % report_interval:
                    print_call(f"Val Acc @ Epoch {epoch_i}: {val_acc}")
                    print_call(f"Val Loss @ Epoch {epoch_i}: {ave_loss}")
                    print_call(f"Best Val Loss was {best_val_loss} @ Epoch {min_loss_epoch}")
                    
                    print_call(self.score(test_x, test_y))
                    sys.stdout.flush()

                curr_lr = self.opt.state_dict()['param_groups'][0]['lr']
                scheduler.step(ave_loss)
                if curr_lr != self.opt.state_dict()['param_groups'][0]['lr']:
                    print_call("Stepped LR")
                    lr_steps += 1
                if ave_loss < best_val_loss:
                    best_sd = self.mlp.state_dict()
                    best_val_loss = ave_loss
                    min_loss_epoch = epoch_i

                    torch.save(best_sd, os.path.join(exp_save_dir, 'checkpoint.bin'))

            epoch_i += 1
        self.mlp.load_state_dict(best_sd)

    def score(self, test_x, test_y, bs=1024):
        with torch.no_grad():
            self.mlp.eval()
            x = torch.Tensor(test_x)
            n_train = x.shape[0]
            y = torch.LongTensor(self.cast_targets(test_y))
            dataset = torch.utils.data.TensorDataset(x, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
            criterion = torch.nn.CrossEntropyLoss()

            class_seen = Counter()
            class_correct = Counter()
            for bi, (bx, by) in enumerate(loader):
                bx = bx.cuda()
                by = by.cuda()
                N, num_channels = bx.shape
                bx = bx.view(N, -1, self.in_c)  # (N, t, in_c)

                # (N, t*in_c*reg_c) if keep_reg else  (N, t*in_c)
                bx = bx.reshape(N, -1) if self.keep_reg else bx[:, :, :-self.reg_c].reshape(N, -1)

                # if not bi%20: print(f"{num_seen} / {n_train}")
                preds = self.mlp(bx)
                preds = preds.argmax(dim=1)

                batch_seen = Counter(by.cpu().numpy())
                batch_correct = {}
                for c in batch_seen:
                    class_mask = by == c
                    batch_correct[c] = (preds[class_mask] == by[class_mask]).sum().item()

                class_seen.update(batch_seen)
                class_correct.update(batch_correct)

            return get_accuracies(class_seen, class_correct)

    def predict(self, test_x, bs=1024, return_vec=False):
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
                N, num_channels = bx.shape
                bx = bx.view(N, -1, self.in_c)  # (N, t, in_c + extra_c)

                # (N, t*in_c*reg_c) if keep_reg else  (N, t*in_c)
                bx = bx.reshape(N, -1) if self.keep_reg else bx[:, :, :-self.reg_c].reshape(N, -1)
                output = self.mlp(bx)
                if not return_vec: output = output.argmax(dim=1)
                preds_list.append(output)


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
        if 'climate' in data_prep_list:
            x_climate_train = np.tile(climate_train, (1, t))  # (N, t*19)
            x_climate_train = x_climate_train.reshape((train_x.shape[0], t, -1))  # (N, t, 19)
            train_x = np.concatenate((train_x, x_climate_train), axis=-1)  # (N, t, c+19)

            x_climate_test = np.tile(climate_test, (1, t))  # (Nte, t*19)
            x_climate_test = x_climate_test.reshape((test_x.shape[0], t, -1))  # (Nte, t, 2)
            test_x = np.concatenate((test_x, x_climate_test), axis=-1)  # (Nte, t, c+19)
        if 'coords' in data_prep_list:
            x_coords_train = np.tile(coords_train, (1, t))  # (N, t*2)
            x_coords_train = x_coords_train.reshape((train_x.shape[0], t, -1))  # (N, t, 2)
            train_x = np.concatenate((train_x, x_coords_train), axis=-1)  # (N, t, c+2)

            x_coords_test = np.tile(coords_test, (1, t))  # (Nte, t*2)
            x_coords_test = x_coords_test.reshape((test_x.shape[0], t, -1))  # (Nte, t, 2)
            test_x = np.concatenate((test_x, x_coords_test), axis=-1)  # (Nte, t, c+2)
        if 'normalize' in data_prep_list:
            print(train_x.mean(axis=(0,1)), train_x.std(axis=(0,1)))
            print(train_x.min())
            # from collections import Counter
            # print(Counter(train_x.reshape(-1)))
            train_clouds = train_x==0
            test_clouds = test_x==0
            train_x[train_clouds] = np.nan
            test_x[test_clouds] = np.nan
            train_mean = np.nanmean(train_x, axis=(0,1))
            train_std = np.nanstd(train_x, axis=(0,1))

            train_x = (train_x - train_mean) / train_std
            test_x =  (test_x  - train_mean) / train_std

            train_x[train_clouds] = 0
            test_x[test_clouds] = 0
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

        _, t, in_c = train_x.shape
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
        elif 'softdtw_centroid' in clf_str:  # might have _transducitve at front
            clf = DTWBarycenter('soft')
        elif 'SGdtw_centroid' in clf_str:  # might have _transducitve at front
            clf = DTWBarycenter('SG')
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
            clf = TransformerNN(in_channels=in_c, wd=args.weight_decay)
        elif clf_str == 'transformer_correlation':
            assert 'coords' in data_prep_list or 'climate' in data_prep_list
            print("ATTENTION: Right now coords is needed in data prep list to give targets but it NOT used as input")
            reg_c = in_c - c  # (9 + x - 9)
            print(f"Regression channels: {reg_c}")
            clf = TransformerCorrelation(args.weight, reg_c, False, in_channels=in_c-reg_c, wd=args.weight_decay)
            print(f"Weight decay: {clf.wd}")
        elif clf_str == 'transformer_correlation_input':
            assert 'coords' in data_prep_list or 'climate' in data_prep_list
            print("ATTENTION: Right now coords/climate is used as input and target")
            reg_c = in_c - c  # (9 + x - 9)
            clf = TransformerCorrelation(args.weight, reg_c, True, in_channels=in_c, wd=args.weight_decay)
            print(f"Weight decay: {clf.wd}")
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
            clf = NTK(args.num_ntk_nets,
                      norm_indivs=args.norm_indivs,
                      greedy_similarity=args.greedy_similarity,
                      dimension=args.dimension,
                      optimize_distributional_average=False)
        elif clf_str == 'ntk_distributional':
            clf = NTK(args.num_ntk_nets,
                      norm_indivs=args.norm_indivs,
                      greedy_similarity=args.greedy_similarity,
                      dimension=args.dimension,
                      optimize_distributional_average=True)
        else:
            raise NotImplementedError

        if args.checkpoint is None:
            sample_weights = None
            if args.sample_weight:
                print('Using class inverse frequency for sample weights')
                classes, counts = np.unique(train_y, return_counts=True)
                inv_counts = 1/counts
                inv_freq = inv_counts/sum(inv_counts)
                class_weights = np.zeros(int(max(classes) + 1), dtype=np.float32)
                for c, inv_f in zip(classes, inv_freq):
                    class_weights[int(c)] = inv_f
                sample_weights = class_weights[train_y.astype(np.long)]
                print(sample_weights)

            clf.fit(train_x, train_y, sample_weights=sample_weights)
        else:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")

            state_dict = torch.load(args.checkpoint, map_location=device)
            clf.mlp.load_state_dict(state_dict)
        # gen_score = clf.transductive_score(train_x, train_y, test_x, test_y) \
        #     if 'transductive' in clf_str else clf.score(test_x, test_y)
        # print(clf_str, data_prep, gen_score)
        # metrics.plot_confusion_matrix(clf, test_x, test_y)
        # plt.savefig(f"{generalization_region}_{clf_str}_{data_prep}.png")
        # pred_test_y = clf.predict(test_x)
        # confusion_matrix = metrics.confusion_matrix(test_y, pred_test_y, labels=interest_classes)
        # np.save(f"{generalization_region}_{clf_str}_{data_prep}.npy", confusion_matrix)
        print(f'{clf_str} {data_prep} scores:')
        scores = clf.score(test_x, test_y)
        for score_str, score in scores.items():
            print(f"{score_str}: {score}")
        print()
        sys.stdout.flush()
