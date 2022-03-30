import os
import ast
import json
import pickle
import argparse
import numpy as np
import pandas as pd

from collections import defaultdict

from interest_classes import interest_classes


BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
CLIMATE_BANDS = ['bio' + (f'0{n}' if n < 10 else f'{n}') for n in range(1, 20)]


def preprocess_separate_bands(paths_to_csvs, out_path):
    all_bands_df = None
    for i, p in enumerate(paths_to_csvs):
        df = pd.read_csv(p)

        # Next two lines only needed on raw dfs (straight from EE)
        df['coords'] = df['.geo'].apply(lambda v: tuple(json.loads(v)['coordinates']))
        df.drop(['system:index', '.geo'], axis=1, inplace=True)

        if all_bands_df is None:
            all_bands_df = df
        else:
            all_bands_df = all_bands_df.merge(df, on='coords',)  # Should  not need suffixes

    # all_bands_df.reset_index(drop=True, inplace=True)

    all_bands_df.to_csv(out_path, index=False)
    return all_bands_df


def preprocess_dates(paths_to_csvs, out_path):
    time_df = None
    for i, p in enumerate(paths_to_csvs):
        df = pd.read_csv(p)
        if 'coords' not in df.columns:
            df['coords'] = df['.geo'].apply(lambda v: str(tuple(json.loads(v)['coordinates'])))
            df.drop(['system:index', '.geo'], axis=1, inplace=True)

        if time_df is None:
            time_df = df
        else:
            time_df = time_df.merge(df, on='coords', suffixes=(None, 'y'))

            for b in df.columns:
                if b.startswith('B'):
                    if i < 2:
                        time_df[b] = time_df[[b, b + 'y']].apply(list, axis=1)
                    else:
                        time_df[[b, b + 'y']].apply(lambda bb: bb[0].append(bb[1]), axis=1)

                    time_df.drop(b + 'y', axis=1, inplace=True)

    # time_df.reset_index(drop=True, inplace=True)

    time_df.to_csv(out_path, index=False)
    return time_df


def preprocess_labels(time_df, path_to_gt, out_path):
    gt = pd.read_csv(path_to_gt)

    gt['coords'] = gt['.geo'].apply(lambda v: str(tuple(json.loads(v)['coordinates'])))
    gt.drop(['system:index', '.geo'], axis=1, inplace=True)

    time_df = time_df.merge(gt, how='left', on='coords')
    # time_df.rename({'cropland': 'y'}, inplace=True)

    time_df.dropna(inplace=True)
    time_df.reset_index(drop=True, inplace=True)

    time_df.to_csv(out_path, index=False)
    return time_df


def preprocess_climate(time_df, path_to_climate, out_path):
    climate = pd.read_csv(path_to_climate)

    climate['coords'] = climate['.geo'].apply(lambda v: str(tuple(json.loads(v)['coordinates'])))
    climate.drop(['system:index', '.geo'], axis=1, inplace=True)

    time_df = time_df.merge(climate, how='left', on='coords')

    time_df.dropna(inplace=True)
    time_df.reset_index(drop=True, inplace=True)

    time_df.to_csv(out_path, index=False)
    return time_df


def collect_band_paths(data_dir):
    org_paths = defaultdict(lambda: defaultdict(list))

    dir_path = os.path.abspath(data_dir)
    paths = os.listdir(data_dir)
    for p in paths:
        if 'landsat' in p:
            _, year, mo, day, band = p.split('_')
            org_paths[year][f'{mo}-{day}'].append(os.path.join(dir_path, p))

        elif 'cdl' in p:
            _, year = p.split('_')
            org_paths[year]['cdl'] = os.path.join(dir_path, p)

    return org_paths


def tile_points(coords, n=2):
    lon_min, lon_max = np.min(coords[:, 0]), np.max(coords[:, 1])
    lat_min, lat_max = np.min(coords[:, 1]), np.max(coords[:, 1])

    lon_cuts = [lon_min]
    lat_cuts = [lat_min]
    for i in range(n-1):
        qth_percentile = (i+1)/n * 100
        lon_cuts.append(np.percentile(coords[:, 0], qth_percentile))
        lat_cuts.append(np.percentile(coords[:, 1], qth_percentile))
    lon_cuts.append(lon_max)
    lat_cuts.append(lat_max)

    lon_intervals = list(zip(lon_cuts, lon_cuts[1:]))
    lat_intervals = list(zip(lat_cuts, lat_cuts[1:]))

    grid_groups = defaultdict(list)
    for i, c in enumerate(coords):
        group = [0, 0]
        for lo, (lon_a, lon_b) in enumerate(lon_intervals):
            if lon_a <= c[0] <= lon_b:
                group[0] = lo
                break

        for la, (lat_a, lat_b) in enumerate(lat_intervals):
            if lat_a <= c[1] <= lat_b:
                group[1] = la
                break

        grid_groups[tuple(group)].append(i)

    return {g: np.array(inds) for g, inds in grid_groups.items()}


def tile_df(full_df, n=2):
    coords = np.array([ast.literal_eval(c) for c in full_df['coords'].values])

    groups = tile_points(coords, n=n)

    for g, inds in groups.items():
        gdf = full_df.iloc[inds]
        gdf.reset_index(drop=True, inplace=True)
        groups[g] = gdf

    return groups


def df_to_array(df, class_filter=False, cloud_filter=False):
    X = df[BANDS].values
    X = np.array([[ast.literal_eval(ts) for ts in row] for row in X])  # (N, c, t)

    coords = df['coords'].values
    coords = np.array([ast.literal_eval(lon_lat) for lon_lat in coords])

    y = np.array(df['cropland'].values)

    climate = np.array(df[CLIMATE_BANDS].values)

    def apply_mask(mask, *arrs):
        return [arr[mask] for arr in arrs]

    # if class_filter:
    #     interest_mask = np.isin(y, interest_classes)
    #     X, y, coords, climate = apply_mask(interest_mask, X, y, coords, climate)

    if cloud_filter:
        cloud_mask = np.any(X.reshape(X.shape[0], -1) > 0, axis=-1)  # (N,)
        X, y, coords, climate = apply_mask(cloud_mask, X, y, coords, climate)

    return X, y, coords, climate


def passed_arguments():
    parser = argparse.ArgumentParser(description='Script to preprocess grid csv data from EE')
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help="Path to directory with all the csvs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = passed_arguments()

    abs_path = os.path.abspath(args.data_path)
    out_path = os.path.join(abs_path, f'landsat_2017.csv')

    ## Once df is already created
    df = pd.read_csv(out_path)

    n = 2
    df_groups = tile_df(df, n)
    for g, gdf in df_groups.items():
        gstring = f'g{int(g[0]*n + g[1])}'
        print(g, gstring)

        X, y, coords, climate = df_to_array(gdf, class_filter=False, cloud_filter=True)

        dest_dir = os.path.join(abs_path, 'usa_' + gstring)
        os.makedirs(dest_dir, exist_ok=True)

        with open(os.path.join(dest_dir, 'values.pkl'), 'wb') as f:
            pickle.dump(X, f)
        with open(os.path.join(dest_dir, 'coords.pkl'), 'wb') as f:
            pickle.dump(coords, f)
        with open(os.path.join(dest_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(y, f)
        with open(os.path.join(dest_dir, 'climate.pkl'), 'wb') as f:
            pickle.dump(climate, f)

        gdf.to_csv(os.path.join(dest_dir, f'landsat_2017_{gstring}.csv'), index=False)

    # org_paths = collect_band_paths(args.data_path)
    # for year, year_data in org_paths.items():
    #     band_df = None
    #
    #     date_df_paths = []
    #     for key, info in year_data.items():
    #         if type(info) is list:
    #             mo, day = key.split('-')
    #             out_path = os.path.join(abs_path, f'landsat_{year}_{mo}_{day}.csv')
    #
    #             preprocess_separate_bands(info, out_path)
    #
    #             date_df_paths.append(out_path)
    #
    #     out_path = os.path.join(abs_path, f'landsat_{year}.csv')
    #     time_df = preprocess_dates(date_df_paths, out_path)
    #
    #     if 'cdl' in year_data:
    #         path_to_gt = year_data['cdl']
    #
    #         time_df = preprocess_labels(time_df, path_to_gt, out_path)

    ## Run below to start from separate dates
    # date_paths = [os.path.join(abs_path, f'landsat_2017_{mo}_{day}.csv') for mo, day in
    #               [('04', '07'), ('04', '23'), ('05', '09'), ('05', '25'),
    #                ('06', '10'), ('06', '26'), ('07', '12'), ('07', '28')]]
    # time_df = preprocess_dates(date_paths, out_path)
    # time_df = preprocess_labels(time_df, os.path.join(abs_path, 'cdl_2017.csv'), out_path)
    # time_df = preprocess_climate(time_df, os.path.join(abs_path, 'climate.csv'), out_path)

    # time_df = pd.read_csv(os.path.join(abs_path, 'landsat_2017_x.csv'))
    # time_df = preprocess_labels(time_df, os.path.join(abs_path, 'cdl_2017.csv'), out_path)


