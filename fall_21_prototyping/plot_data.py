import os
import torch
import pickle
import argparse
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from interest_classes import interest_classes, interest_class_names


def mercator(coords, w, h):
    x = (coords[:, 0] + 180) * (w/360)
    lat_rad = coords[:, 1] * np.pi / 180
    mer = np.log(np.tan(lat_rad/2 + np.pi/4))
    y = (w * mer/(2 * np.pi)) - (h/2)
    return np.stack((y, x))


def proc_transformer_loss_feats(transformer, X, y, batch_size=1024, get_feats=False):
    transformer.eval()

    XT = torch.from_numpy(X.reshape(X.shape[0], -1).astype(np.float32))

    orig_class_to_index = {int(c): i for i, c in enumerate(interest_classes)}
    yt = torch.LongTensor([orig_class_to_index[int(y_i)] for y_i in y])
    # yt = torch.zeros(len(y), dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(XT, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    losses = torch.empty(0, dtype=torch.float32, requires_grad=False)
    feats = torch.empty(0, dtype=torch.float32, requires_grad=False)
    with torch.no_grad():
        for i, (bx, by) in enumerate(loader):
            out = transformer(bx, return_final_feature=get_feats)
            # class_mask = (by[..., None] == ar2).any(-1)

            if get_feats:
                pred, feat = out
                feats = torch.cat((feats, feat))
            else:
                pred = out

            batch_losses = loss_fn(pred, by)
            losses = torch.cat((losses, batch_losses))

            if i % 10 == 0:
                print(f"Loss at iter {i}: {round(batch_losses.mean().item(), 4)}")

    if get_feats:
        return losses.numpy(), feats.numpy()  # (N,), (N, d)
    return losses.numpy()


def proc_nearest_feats(feats, target_inds, source_inds, k=100):
    target_feats = feats[target_inds]  # (Nt, d)
    source_feats = feats[source_inds]  # (Ns, d)

    # dist = cdist(source_feats, target_feats)  # (Ns, Nt)
    # nearest = np.argmax(-dist, axis=-1)  # (Ns,)
    # source_anchors = anchors[nearest]  # (Ns, )

    dist = cdist(target_feats, source_feats)  # (Nt, Ns)
    # nearest = np.argmax(-dist, axis=-1)  # (Nt,)
    nearest = np.argsort(dist, axis=-1)[:, :k]  # (Nt, k)

    return nearest


def get_transformer_loss_feats(checkpoint_dir, X, y, target_inds, source_inds, get_feats=True):
    from transformer import Transformer
    transformer = Transformer(len(interest_classes), 9, 2)
    sd = torch.load(os.path.join(checkpoint_dir, 'checkpoint.bin'), map_location='cpu')
    transformer.load_state_dict(sd)

    feats_file = os.path.join(checkpoint_dir, 'features.pkl')
    losses_file = os.path.join(checkpoint_dir, 'losses.pkl')
    nearest_feats_file = os.path.join(checkpoint_dir, 'nearest_features.pkl')
    if os.path.isfile(feats_file) and os.path.isfile(losses_file):
        print(f"Loading features from {feats_file}, losses from {losses_file}")
        with open(losses_file, 'rb') as f:
            losses = pickle.load(f)
        with open(feats_file, 'rb') as f:
            feats = pickle.load(f)
    else:
        losses, feats = proc_transformer_loss_feats(transformer, X, y, get_feats=get_feats)
        with open(losses_file, 'wb') as f:
            pickle.dump(losses, f)
        with open(feats_file, 'wb') as f:
            pickle.dump(feats, f)

    if os.path.isfile(nearest_feats_file):
        print(f"Loading nearest neighbour indices from {nearest_feats_file}")
        with open(nearest_feats_file, 'rb') as f:
            nearest_feats = pickle.load(f)
    else:
        nearest_feats = proc_nearest_feats(feats, target_inds, source_inds,)
        with open(nearest_feats_file, 'wb') as f:
            pickle.dump(nearest_feats, f)

    return transformer, losses, feats, nearest_feats


def plot_transformer_loss(losses, coords,):

    losses = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))

    cmap = get_cmap('viridis', 100)
    colors = cmap(losses)

    merc = mercator(coords, 200, 100)
    pix = merc.T  # (N, 2)
    plot = plt.scatter(pix[:, 1], pix[:, 0], c=losses, cmap=cmap)

    plt.colorbar(plot)
    plt.show()


def plot_transformer_feat(nearest, coords, target_inds, source_inds, k=20):
    pix = mercator(coords, 200, 100).T  # (N, 2)

    # anchors = np.arange(target_feats.shape[0])
    # anchors = anchors / np.max(anchors)
    # cmap = get_cmap('viridis', len(anchors))
    # # target_colors = cmap(anchors)  # (Nt, 4)
    #
    # dist = cdist(source_feats, target_feats)  # (Ns, Nt)
    # nearest = np.argmax(-dist, axis=-1)  # (Ns,)
    # source_anchors = anchors[nearest]  # (Ns, )

    source_anchors = np.arange(source_inds.shape[0])
    source_anchors = source_anchors / np.max(source_anchors)
    cmap = get_cmap('RdYlGn', len(source_anchors))
    # cmap = get_cmap('viridis', len(source_anchors))

    target_anchors = np.mean(source_anchors[nearest[:, :k]], axis=-1)  # (Nt,)

    source_pix = pix[source_inds]
    plt.scatter(source_pix[:, 1], source_pix[:, 0], c=source_anchors, cmap=cmap, s=1, alpha=1)

    target_pix = pix[target_inds]
    plt.scatter(target_pix[:, 1], target_pix[:, 0], c=target_anchors, cmap=cmap, s=1, alpha=1)

    plt.show()

    return nearest


def load_data_from_pickle(path_dir):
    with open(os.path.join(path_dir, 'values.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(path_dir, 'labels.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(path_dir, 'coords.pkl'), 'rb') as f:
        coords = pickle.load(f)
    return x, y, coords


def passed_args():
    parser = argparse.ArgumentParser(description="Plot stuff")
    parser.add_argument('--data_path', type=str, default='../data_grid/grid_10000', help='Path to data')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint dir')
    return parser.parse_args()


if __name__ == "__main__":
    args = passed_args()

    regions = sorted([f for f in os.listdir(args.data_path) if f.startswith('usa')])
    source_regions = [regions[0], regions[1], regions[2]]
    target_regions = [regions[3]]

    X, Y, coords = [], [], []
    last_ind, source_inds, target_inds = 0, [], []
    for r in regions:
        x, y, coord = load_data_from_pickle(os.path.join(args.data_path, r))

        inds = list(range(last_ind, last_ind + x.shape[0]))
        if r in source_regions:
            source_inds.extend(inds)
        else:
            target_inds.extend(inds)
        last_ind += x.shape[0]

        X.append(x)
        Y.append(y)
        coords.append(coord)

        print(f"Region {r} shape: {x.shape}")

    X = np.concatenate(X, axis=0)  # (N, c, t)
    Y = np.concatenate(Y, axis=0)  # (N,)
    coords = np.concatenate(coords, axis=0)  # (N, 2)  fmt: (lon, lat)
    orig_source_inds = np.array(source_inds)  # (Ns,)
    target_inds = np.array(target_inds)  # (Nt,)
    print(X.shape)

    # Sort for pretty display
    source_coords = coords[orig_source_inds]
    sorted_coords = np.lexsort((source_coords[:, 1], source_coords[:, 0]))  # (Sort by lat, then by lon)
    source_inds = orig_source_inds[sorted_coords]
    # X[source_inds] = X[source_inds][sorted_coords]
    # Y[source_inds] = Y[source_inds][sorted_coords]
    # coords[source_inds] = coords[source_inds][sorted_coords]

    transformer, losses, feats, nearest = get_transformer_loss_feats(
        args.checkpoint, X, Y, target_inds, source_inds, get_feats=True
    )

    # lat_lon = np.lexsort((source_coords[:, 0], source_coords[:, 1]))
    # lat_lon_inds = orig_source_inds[lat_lon]
    # map = np.lexsort((coords[lat_lon_inds][:, 1], coords[lat_lon_inds][:, 0]))
    # assert np.all(lat_lon_inds[map] == source_inds)
    # print(map)
    #
    # # feats[lat_lon_inds] = feats[source_inds]
    # nearest = map[nearest]
    plot_transformer_feat(nearest, coords, target_inds, source_inds, k=1)


