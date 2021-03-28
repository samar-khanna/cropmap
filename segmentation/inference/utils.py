import numpy as np
from PIL import Image, ImageDraw

from utils.colors import get_color_choice


def bytescale(img, high=255):
    """
    Converts an image of arbitrary int dtype to an 8-bit (uint8) image.
    @param img: (h, w, #c) numpy array of any int dtype
    @param high: Max pixel value
    @return: (h, w, #c) numpy array of type uint8
    """
    # Find min and max across each channel
    im = img.reshape(img.shape[-1], -1)
    im_min = np.min(im, axis=1)
    im_max = np.max(im, axis=1)

    scale = 255 / (im_max - im_min)
    im_arr = (img - im_min) * scale

    return im_arr.astype(np.uint8)


def draw_mask_on_im(img, masks):
    """
    Helper method that opens an image, draws the segmentation masks in `masks`
    as bitmaps, and then returns the masked image.
    @param img: np array of shape (h, w, #c)
    @param masks: np array of shape: (#c, h, w)
    @return: image with mask drawn on it as well as raw mask
    """
    # Open the image and set up an ImageDraw object
    im = Image.fromarray(img, mode='RGB')
    im_draw = ImageDraw.Draw(im)

    # Draw the masks in a separate image as well
    raw_mask = Image.fromarray(np.zeros(img.shape[0:2])).convert('RGB')
    mask_draw = ImageDraw.Draw(raw_mask)

    # Draw the bitmap for each class (only if class mask not empty)
    for i, mask in enumerate(masks):
        if mask.any():
            # Create mask, scale its values by 64 so that opacity not too low/high
            mask_arr = mask.astype(np.uint8) * 64
            mask_im = Image.fromarray(mask_arr, mode='L')

            # Get color choice, and draw on image and on raw_mask
            color = get_color_choice(i)
            im_draw.bitmap((0, 0), mask_im, fill=color)
            mask_draw.bitmap((0, 0), mask_im, fill=color)

    return im, raw_mask
