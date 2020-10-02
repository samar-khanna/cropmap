import numpy as np


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
                f"Number of channels to drop {num_to_drop} not equal to expected - actual {C-c}"

            return x[self.channels_to_drop], y

        return x[:c, ...], y