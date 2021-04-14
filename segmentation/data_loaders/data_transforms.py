import numpy as np
import torch


class MaskCloudyTargetsTransform:
    def __init__(self, mask_value=-1, cloud_value=0., is_conservative=False):
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

    def __call__(self, sample):
        xs, y = sample

        if not isinstance(xs, list):
            xs = [xs]

        invalid_mask = np.zeros(xs[0].shape[1:], dtype=np.bool)  # shape (h, w)
        for t, x_t in enumerate(xs):

            invalid = ~np.any(x_t != self.cloud_value, axis=0)  # shape (h, w)
            if t == 0 or self.is_conservative:
                invalid_mask = invalid_mask | invalid
            else:
                invalid_mask = invalid_mask & invalid

        y[..., invalid_mask] = self.mask_value  # shape (c, h, w)

        return xs, y


class MaskBandTransform:
    def __init__(self, bands_to_mask, mask_values=[]):
        """
        Creates a transform to mask the bands of the input image with a specified value.
        Expects to be called on a `numpy array`.\n
        Requires:\n
          `bands_to_mask`: List of indices (must be zero-indexed) specifiying the bands
          in the input array that will be masked.\n
          `mask_values`: List of numbers specifying the value with which to mask each band.
          If not specified for all bands, then must be same length as `bands_to_mask`.
          If empty, will use the average value of the band in that tile.
        """
        super().__init__()
        self.bands_to_mask = np.array(bands_to_mask, dtype=np.int)
        self.mask_values = np.array(mask_values, dtype=np.float)

    def __call__(self, sample):
        """
        Masks the bands of the input image.
        Requires: \n
          `sample`: `(x, y)` where `x` is a numpy array of shape (#bands, h, w)
          and `y` is a numpy array of shape (c, h, w)
        """
        x, y = sample
        num_bands, h, w = x.shape
        num_mask_values = len(self.mask_values)

        if num_mask_values == num_bands:
            # Index the interested masks, length is now #bands_to_mask
            mask_fill = self.mask_values[self.bands_to_mask]
        elif num_mask_values == 0:
            # Reduce to a mean value per band (local averaging)
            mask_fill = np.mean(np.mean(x, axis=-1), axis=-1)
            mask_fill = mask_fill[self.bands_to_mask]
        else:
            assert num_mask_values == len(self.bands_to_mask), \
                f"Number of mask values ({num_mask_values}) is not 0 or {num_bands}, " + \
                "so must be same length as bands_to_mask."
            mask_fill = self.mask_values

        # Create the mask for the interested bands
        # 1) Make an inverted array of shape (w, h, #bands to mask)
        # 2) Fill that array with the mask values (for each 2d slice)
        # 3) Reshape to (#bands, h, w)
        if len(self.bands_to_mask) > 0:
            mask = np.ones((w, h, len(self.bands_to_mask)))
            mask = mask * mask_fill
            mask = mask.T

            x[self.bands_to_mask] = mask

        return x, y


class DropChannelTransform:
    def __init__(self, expected_num_channels, channels_to_drop=[]):
        """
        Creates a transform to drop extra channels from the input image so that the
        input image matches the expected input size for the model's first layer.
        It will not augment the input with dummy channels if the model expects a larger
        input image than what is provided. \n
        Warning: use carefully, because if model size and input size don't match, won't
        stop execution. \n
        Requires:\n
          `expected_num_channels`: Integer expected #channels of input image.
          `channels_to_drop`: List of channel indices ranging from 0 through #num_channels
             of the actual input tile. It's length must be `C - c`, where `C` is the actual number
             of input channels in the tile, and `c` is the expected number.
            If empty, will just drop the last channels of the input tile until the number of
            channels match.
        """
        super().__init__()
        self.c = expected_num_channels
        self.channels_to_drop = np.array(channels_to_drop, dtype=np.int)

    def __call__(self, sample):
        """
        Drops the specified channels of the input image.
        Requires:\n
          `sample`: `(x, y)` where `x` is a numpy array of shape (#bands, h, w)
          and `y` is a numpy array of shape (c, h, w)
        """
        x, y = sample
        num_channels, h, w = x.shape

        # Early bail out, return identity
        if num_channels == self.c:
            return x, y

        c, C = self.c, num_channels

        assert C >= c, \
            f"Expected #channels {C} is greater than total #channels {c}"

        if len(self.channels_to_drop) > 0:
            num_to_drop = len(self.channels_to_drop)
            assert num_to_drop == C - c, \
                f"Number of channels to drop {num_to_drop} not equal to expected - actual {C - c}"

            return x[self.channels_to_drop], y

        return x[:c, ...], y


class PixelStaticTransform:
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, sample):
        if isinstance(sample, tuple): raise NotImplementedError  # Just need to add xy parsing logic
        # TODO: parallelize this instead of iterating in for loop

        if isinstance(sample, list):
            # timeseries
            for i, s in enumerate(sample):
                sample[i] = self(sample[i])
        else:
            for channel_i in range(sample.shape[1]):
                channel_std = sample[:, channel_i].std()
                sample[:, channel_i] += self.scale * channel_std * np.random.randn(*sample[:, channel_i].shape)
        return sample


class HorizontalFlipSimCLRTransform:
    def __init__(self):
        super().__init__()

    def __call__(self, images, do_flips):
        """
        do_flips is boolean list of length of images
        """
        assert len(images) == len(do_flips)
        for i, do_flip in enumerate(do_flips):
            if do_flip: images[i] = images[i].flip([-2])
        return images


class VerticalFlipSimCLRTransform:
    def __init__(self):
        super().__init__()

    def __call__(self, images, do_flips):
        """
        do_flips is boolean list of length of images
        """
        assert len(images) == len(do_flips)
        for i, do_flip in enumerate(do_flips):
            if do_flip: images[i] = images[i].flip([-1])
        return images


class RotationSimCLRTransform:
    def __init__(self):
        super().__init__()

    def __call__(self, images, rotations, inverse=False):
        assert len(images) == len(rotations)
        for i, rot in enumerate(rotations):
            rot = 4 - rot if inverse else rot
            images[i] = images[i].rot90(rot, dims=[-2, -1])
        return images


class RandomResizedCropSimCLRTransform:
    def __init__(self, minimum_scale=0.5):
        # Have minimum crop scale low for now because we have a constant resolution
        super().__init__()
        self.minimum_scale = minimum_scale
        assert self.minimum_scale > 0 and self.minimum_scale < 1

    def __call__(self, images, crop_seeds):
        """
        crop seeds are triples that dictate 
        1) The scale to sample at (uniform between (minimum_scale, 1)
        2) The location to sample at given the scale
        """
        orig_resolution = images.shape[-1]
        assert images.shape[-2] == orig_resolution, "We expect square images, probably not a big deal"
        for i, (image, seed_triple) in enumerate(zip(images, crop_seeds)):
            new_crop_scale = seed_triple[0] * (1 - self.minimum_scale) + self.minimum_scale
            new_crop_resolution = int(orig_resolution * new_crop_scale)
            # Corners can be indices anywhere from 0 to (orig_resolution - new_crop_resolution)
            corner_range = orig_resolution - new_crop_resolution
            top = int(seed_triple[1] * corner_range)
            left = int(seed_triple[2] * corner_range)
            raw_crop = image[:, top:top + new_crop_resolution, left:left + new_crop_resolution]
            resized_crop = torch.nn.functional.interpolate(raw_crop.unsqueeze(0),
                                                           size=(orig_resolution, orig_resolution),
                                                           mode='bilinear').squeeze()
            images[i] = resized_crop
        return images
