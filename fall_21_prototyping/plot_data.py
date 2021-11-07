import os
import torch
import pickle
import argparse
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib
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

    cmap = get_cmap('viridis', 1000)
    colors = cmap(losses)

    merc = mercator(coords, 200, 100)
    pix = merc.T  # (N, 2)
    plot = plt.scatter(pix[:, 1], pix[:, 0], c=losses, cmap=cmap, s=1)

    plt.colorbar(plot)
    # plt.show()
    plt.savefig('tmp.png')

def coords_to_colors(coords):
    N = coords.shape[0]
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    min_arr = np.array([min_x, min_y])
    max_arr = np.array([max_x, max_y])
    mid_arr = np.array([mid_x, mid_y])
    # these are coords in [-1,1]^2 box
    normed_coords = (coords - mid_arr) / (0.5 * (max_arr - min_arr))
    # print(normed_coords.min(axis=0), normed_coords.max(axis=0))
    # want to calculate r and theta for
    raw_r = np.sqrt( (normed_coords**2).sum(axis=1))
    normed_r = raw_r / raw_r.max()
    theta = (np.arctan2(normed_coords[:,1], normed_coords[:,0]) / (2 * np.pi)) + 0.5
    print(normed_r.min(), normed_r.max(), theta.min(), theta.max())
    # hsv_vals = np.concatenate([normed_coords[:,:1], 0.8 * np.ones((N, 1)), normed_coords[:,1:]], axis=1)
    hsv_vals = np.stack([theta, 1 * np.ones(N), normed_r], axis=1)
    return matplotlib.colors.hsv_to_rgb(hsv_vals)

def plot_transformer_feat(nearest, coords, target_inds, source_inds, target_region_i, k=1, save_dir='plots/'):
    pix = mercator(coords, 200, 100).T  # (N, 2)

    # anchors = np.arange(target_feats.shape[0])
    # anchors = anchors / np.max(anchors)
    # cmap = get_cmap('viridis', len(anchors))
    # # target_colors = cmap(anchors)  # (Nt, 4)
    #
    # dist = cdist(source_feats, target_feats)  # (Ns, Nt)
    # nearest = np.argmax(-dist, axis=-1)  # (Ns,)
    # source_anchors = anchors[nearest]  # (Ns, )

    # below is 0 to 1 indexing of source
    source_anchors = np.arange(source_inds.shape[0])
    source_anchors = source_anchors / np.max(source_anchors)
    cmap = get_cmap('RdYlGn', len(source_anchors))
    # cmap = get_cmap('viridis', len(source_anchors))

    # these are 0 to 1 of matching color
    # commented out below line b/c not using for for polar heatmap
    # target_anchors = np.mean(source_anchors[nearest[:, :k]], axis=-1)  # (Nt,)

    coord_colors = coords_to_colors(coords)

    source_pix = pix[source_inds]
    # plt.scatter(source_pix[:, 1], source_pix[:, 0], c=source_anchors, cmap=cmap, s=1, alpha=1)
    plt.scatter(source_pix[:, 1], source_pix[:, 0], c=coord_colors[source_inds], s=1, alpha=1)

    target_pix = pix[target_inds]
    # plt.scatter(target_pix[:, 1], target_pix[:, 0], c=target_anchors, cmap=cmap, s=1, alpha=1)
    plt.scatter(target_pix[:, 1], target_pix[:, 0], c=coord_colors[source_inds][nearest[:,0]], s=1, alpha=1)

    # plt.show()
    plt.savefig(f"{save_dir}/knn_transformer_feats_{target_region_i}.png")

def plot_nearest_climate(coords, target_inds, source_inds, climates, target_region_i,
                         normalize=True, save_dir='plots/', k=1):
    normed_climates = StandardScaler().fit_transform(climates) if normalize else climates
    source_climates = normed_climates[source_inds]
    target_climates = normed_climates[target_inds]


    print("Computing climate nearest neighbors")
    dist = cdist(target_climates, source_climates)  # (Nt, Ns)
    nearest = np.argsort(dist, axis=-1)[:, :1]  # (Nt, k)

    pix = mercator(coords, 200, 100).T  # (N, 2)

    coord_colors = coords_to_colors(coords)

    source_pix = pix[source_inds]
    plt.scatter(source_pix[:, 1], source_pix[:, 0], c=coord_colors[source_inds], s=1, alpha=1)

    target_pix = pix[target_inds]
    plt.scatter(target_pix[:, 1], target_pix[:, 0], c=coord_colors[source_inds][nearest[:,0]], s=1, alpha=1)

    # plt.show()
    normalization_str = "normed" if normalize else "raw"
    plt.savefig(f"{save_dir}/knn_climate_{normalization_str}_{target_region_i}.png")

def plot_nearest_x(coords, target_inds, source_inds, X, target_region_i,
                         normalize=True, save_dir='plots/', k=1):
    # n  x 9 x 8
    # want to eliminate pixels with any amount of cloud
    # First get n x 8 mask of all cloudy
    cloud_mask = np.any(X > 0, axis=1)  # (N, 8) False if ALL are 0 at timepoint
    cloud_mask = np.all(cloud_mask, axis=1) # N  True only if ALL pixels are fully clear
    print(X.shape, cloud_mask.shape, cloud_mask.sum())
    X = X.reshape(X.shape[0], -1)

    normed_X = StandardScaler().fit_transform(X) if normalize else X
    source_X = normed_X[source_inds]
    target_X = normed_X[target_inds]

    print("Computing X data nearest neighbors")
    dist = cdist(target_X, source_X)  # (Nt, Ns)
    nearest = np.argsort(dist, axis=-1)[:, :1]  # (Nt, k)

    pix = mercator(coords, 200, 100).T  # (N, 2)

    coord_colors = coords_to_colors(coords)
    # False in cloud mask = cloudy --> set to black
    coord_colors[~cloud_mask] = (0, 0, 0)

    source_pix = pix[source_inds]
    source_colors = coord_colors[source_inds]
    plt.scatter(source_pix[:, 1], source_pix[:, 0], c=source_colors, s=1, alpha=1)

    target_pix = pix[target_inds]
    target_cloud_mask = cloud_mask[target_inds]
    target_colors = coord_colors[source_inds][nearest[:,0]]
    target_colors[~target_cloud_mask] = (0, 0, 0)
    plt.scatter(target_pix[:, 1], target_pix[:, 0], c=target_colors, s=1, alpha=1)

    # plt.show()
    normalization_str = "normed" if normalize else "raw"
    plt.savefig(f"{save_dir}/knn_X_{normalization_str}_{target_region_i}.png")

    return nearest

def plot_labels(coords, Y, n_classes=None, save_dir='plots/'):
    print(coords.shape, Y.shape)
    pix = mercator(coords, 200, 100).T  # (N, 2)
    coord_colors = Y # coords_to_colors(coords)
    counter = Counter(Y)
    print(counter)
    most_common = counter.most_common(n_classes)
    most_common_classes, most_common_counts = zip(*most_common)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=1200)
    # doing below to see if sequential plotting is making weird
    plotting_points = []
    plotting_colors = []
    cm = plt.get_cmap('gist_rainbow')
    if n_classes is None: n_classes = len(counter.keys())
    colors = [cm(i/n_classes) for i in np.random.permutation(n_classes)]
    for i, (class_i, class_count) in enumerate(most_common):
        class_pix = pix[Y==class_i]
        plotting_points.append(class_pix)
        plotting_colors.extend([colors[i]] * class_pix.shape[0])
    plotting_points = np.concatenate(plotting_points)
    # shuffling so don't layer in deterministic way
    idx = np.random.permutation(len(plotting_points))
    plotting_points = plotting_points[idx]
    plotting_colors = [plotting_colors[i] for i in idx]
    ax.scatter(plotting_points[:, 1], plotting_points[:, 0], s=0.01, color=plotting_colors, alpha=1)
    """
    for class_i, class_count in most_common:
        class_pix = pix[Y==class_i]
        ax.scatter(class_pix[:, 1], class_pix[:, 0], s=1, alpha=0.1, label=class_i)
    plt.legend()
    """
    # plt.tight_layout()
    plt.savefig(f"{save_dir}/labels.png")


climate_band_to_name = {'01':'Annual Mean Temp',
                        '02':'Mean diurnal range',
                        '03':'Isothermality (diurnal range / annual range)',
                        '04':'Temperature seasonality',
                        '05':'Max temp of warmest month',
                        '06':'Min temp of coldest month',
                        '07':'Temp annual range',
                        '08':'Mean temp of wettest quarter',
                        '09':'Mean temp of driest quarter',
                        '10':'Mean temp of warmest quarter',
                        '11':'Mean temp of coldest quarter',
                        '12':'Annual precip',
                        '13':'Precip of wettest month',
                        '14':'Precip of driest month',
                        '15':'Precipitation seasonality (?)',
                        '16':'Precip of wettest quarter',
                        '17':'Precip of driest quarter',
                        '18':'Precip of warmest quarter',
                        '19':'Precip of coldest quarter',
                        }

def plot_all_climate_bands(coords, climates, save_dir='plots/'):
    pix = mercator(coords, 200, 100).T  # (N, 2)
    # climates is N x 19

    cmap = get_cmap('coolwarm')

    for band_i in range(climates.shape[1]):
        print(f"Plotting band {band_i}")
        raw_band = climates[:, band_i]
        band = (raw_band - np.min(raw_band)) / (np.max(raw_band) - np.min(raw_band))
        colors = cmap(band)
        fig, ax = plt.subplots()
        ax.scatter(pix[:, 1], pix[:, 0], c=colors, s=1, cmap=cmap)
        # cbar = plt.colorbar(plot, ticks=[0, 0.5, 1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = plt.colorbar(sm, ticks=[0, 0.5, 1])
        cbar_labels = [raw_band.min(), (raw_band.min() + raw_band.max()) / 2, raw_band.max()]
        cbar.ax.set_yticklabels(cbar_labels)
        band_id = f"{band_i+1:02d}"
        plt.suptitle(f"Band {band_id}")
        plt.title(climate_band_to_name[band_id])
        plt.savefig(f"{save_dir}/climate_band_{band_i}.png")
        plt.close('all')


climate_band_to_name = {'1':'Ultra blue',
                        '2':'Blue',
                        '3':'Green',
                        '4':'Red',
                        '5':'Near IR',
                        '6':'Shortwave IR 1',
                        '7':'Shortwave IR 2',
                        '10':'Brightness temp (?)',
                        '11':'Brightness temp (also?)',
                        }

def plot_timeseries_X_bands(coords, X, save_dir='plots/'):
    pix = mercator(coords, 200, 100).T  # (N, 2)
    # data is N x 9 x 8
    cmap = get_cmap('coolwarm')

    for band_i in range(X.shape[1]): # traversing over 8
        print(f"Plotting band {band_i}")
        raw_band = X[:, band_i, :]
        cloud_mask = raw_band > 0
        raw_clear_band = raw_band[cloud_mask]
        # Want to adjust this for clouds (exclude 0s from min)
        lower_bound = np.quantile(raw_clear_band, 0.1)
        upper_bound = np.quantile(raw_clear_band, 0.9)
        band = (raw_band - lower_bound) / (upper_bound - lower_bound)
        band = np.clip(band, 0, 1)
        colors = cmap(band)
        colors[~cloud_mask] = 0
        print(band.min(), band.max())
        # want points to be alpha 0 (transparent) where there are clouds (0 value)
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        for ax_i in range(8):
            ax = axes[ax_i // 4][ax_i % 4]
            ax.set_axis_off()
            sc = ax.scatter(pix[:, 1], pix[:, 0], c=colors[:, ax_i], s=1)
            if not ax_i:
                band_id = f"{band_i+1 if band_i<7 else band_i+3}"
                ax.set_title(f"Band {band_id}: {climate_band_to_name[band_id]}")
            else:
                ax.set_title(f"Timepoint: {ax_i}")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = plt.colorbar(sm, ticks=[0, 0.5, 1])
        cbar_labels = [raw_clear_band.min(), (raw_clear_band.min() + raw_clear_band.max()) / 2, raw_clear_band.max()]
        cbar.ax.set_yticklabels(cbar_labels)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/X_band_{band_i}.png")
        plt.close('all')


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


def passed_args():
    parser = argparse.ArgumentParser(description="Plot stuff")
    parser.add_argument('--data_path', type=str, default='../data_grid/grid_10000', help='Path to data')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint dir')
    parser.add_argument('--save-dir', type=str, default='plots/', help='Dir to save resulting image(s) to')
    return parser.parse_args()


if __name__ == "__main__":
    args = passed_args()

    regions = sorted([f for f in os.listdir(args.data_path) if f.startswith('usa')])

    target_region_i = int(args.checkpoint.split('transformer_g')[1][0])
    print(target_region_i)
    source_regions = [regions[i] for i in range(4) if i!=target_region_i]
    target_regions = [regions[i] for i in range(4) if i==target_region_i]
    print(source_regions, target_regions)
    # source_regions = [regions[1], regions[2], regions[3]]
    # target_regions = [regions[0]]

    X, Y, coords, climates = [], [], [], []
    last_ind, source_inds, target_inds = 0, [], []
    for r in regions:
        x, y, coord, climate = load_data_from_pickle(os.path.join(args.data_path, r))

        inds = list(range(last_ind, last_ind + x.shape[0]))
        if r in source_regions:
            source_inds.extend(inds)
        else:
            target_inds.extend(inds)
        last_ind += x.shape[0]

        X.append(x)
        Y.append(y)
        coords.append(coord)
        climates.append(climate)

        print(f"Region {r} shape: {x.shape}")

    X = np.concatenate(X, axis=0)  # (N, c, t)
    Y = np.concatenate(Y, axis=0)  # (N,)
    coords = np.concatenate(coords, axis=0)  # (N, 2)  fmt: (lon, lat)
    climates = np.concatenate(climates, axis=0)  # (N, 19)
    orig_source_inds = np.array(source_inds)  # (Ns,)
    target_inds = np.array(target_inds)  # (Nt,)
    print(X.shape, Y.shape, coords.shape, climates.shape)

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
    #
    # plot_transformer_loss(losses, coords)


    ###### IN PROGRESS ####
    plot_labels(coords, Y)
    ###### BELOW THIS LINE IS COMPLETED ####
    # plot_timeseries_X_bands(coords, X)
    # plot_all_climate_bands(coords, climates)
    # plot_nearest_x(coords, target_inds, source_inds, X, target_region_i, save_dir=args.save_dir)
    # plot_nearest_x(coords, target_inds, source_inds, X, target_region_i,
    #                normalize=False, save_dir=args.save_dir)
    # plot_nearest_climate(coords, target_inds, source_inds, climates, target_region_i,
    #                       save_dir=args.save_dir)
    # plot_nearest_climate(coords, target_inds, source_inds, climates, target_region_i,
    #                      normalize=False, save_dir=args.save_dir)
    # plot_transformer_feat(nearest, coords, target_inds, source_inds, target_region_i, k=1, save_dir=args.save_dir)


