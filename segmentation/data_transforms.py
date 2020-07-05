import numpy as np


class MaskBandTransform:
  def __init__(self, bands_to_mask, mask_values=[]):
    """
    Creates a transform to mask the bands of the input image with a specified value.
    Expects to be called on a `numpy array`.
    Requires:
      `bands_to_mask`: List of indices (must be zero-indexed) specifiying the bands
      in the input array that will be masked.
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
    Requires:
      `sample`: `(x, y)` where `x` is a numpy array of shape (#bands, h, w)
      and `y` is a numpy array of shape (c, h, w)
    """
    x, y = sample
    num_bands, h, w = x.shape

    if len(self.mask_values) == num_bands:
      # Index the interested masks, length is now #bands_to_mask
      mask_fill = self.mask_values[self.bands_to_mask]
    elif len(self.mask_values) == 0:
      # Reduce to a mean value per band (local averaging)
      mask_fill = np.mean(np.mean(x, axis=-1), axis=-1)
      mask_fill = mask_fill[self.bands_to_mask]
    else:
      assert len(self.mask_values) == len(self.bands_to_mask),\
        f"Number of mask values is not 0 or {num_bands}, so must be same length as bands_to_mask."
      mask_fill = self.mask_values
      
    # Create the mask for the interested bands
    # 1) Make an inverted array of shape (w, h, #bands to mask)
    # 2) Fill that array with the mask values (for each 2d slice)
    # 3) Reshape to (#bands, h, w)
    mask = np.ones((w, h, len(self.bands_to_mask)))
    mask = mask * mask_fill
    mask = mask.T

    x[self.bands_to_mask] = mask

    return x, y