from segmentation.data_loaders import data_transforms
import torch
import numpy as np
from random import random, randint

hflip = data_transforms.HorizontalFlipSimCLRTransform()
vflip = data_transforms.VerticalFlipSimCLRTransform()
rot = data_transforms.RotationSimCLRTransform()
crop = data_transforms.RandomResizedCropSimCLRTransform()

images_1 = torch.randn(10, 9, 64, 64)
images_2 = images_1.clone()

# Feed model
n = images_1.shape[0]
do_hflips = [[ (random()>0.5) for _ in range(n)] for _ in range(2)]
do_vflips = [[ (random()>0.5) for _ in range(n)] for _ in range(2)]
do_rots = [[randint(0,3) for _ in range(n)] for _ in range(2)]
crop_seeds = np.random.uniform(size=(n, 3))

for i, images in enumerate([images_1, images_2]):
    if i==0: images = crop(images, crop_seeds)
    images = hflip(images, do_hflips[i])
    images = vflip(images, do_hflips[i])
    images = rot(images, do_rots[i])
    if i==0:
        images_1 = images.clone()
    elif i==1:
        images_2 = images.clone()
# Doing linear transform here
features_1 = 5 * images_1.clone() + 2
features_2 = 5 * images_2.clone() + 2

difference_map = (features_1-features_2).abs()
print( difference_map.max(), difference_map.min(), difference_map.mean())
print(features_1.abs().mean())

for i, features in enumerate([features_1, features_2]):
    features = rot(features, do_rots[i], inverse=True)
    features = vflip(features, do_hflips[i])
    features = hflip(features, do_hflips[i])
    if i==0:
        features_1 = features.clone()
    elif i==1:
        features_2 = features.clone()
features_2 = crop(features_2, crop_seeds)

difference_map = (features_1-features_2).abs()
print( difference_map.max(), difference_map.min(), difference_map.mean())
print(features_1.abs().mean())
