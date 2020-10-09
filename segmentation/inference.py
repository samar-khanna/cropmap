import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import logging
import argparse
import numpy as np
from PIL import Image, ImageDraw
from utils.colors import get_color_choice
from metrics import create_metrics_dict, calculate_metrics, MeanMetric
from segmentation import load_model, save_model, create_dataset, ConfigHandler


def draw_mask_on_im(img, masks):
    """
    Helper method that opens an image, draws the segmentation masks in `masks`
    as bitmaps, and then returns the masked image.
    Requires:
        `img`: np array of shape (h, w, #c)
        `masks`: np array of shape: (#c, h, w)
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


def bytescale(img, high=255):
    """
    Converts an image of arbitrary int dtype to an 8-bit (uint8) image.
    Requires:
    `img`: (h, w, #c) numpy array of any int dtype
    Returns:
    `im_arr`: (h, w, #c) numpy array of type uin8
    """
    # Find min and max across each channel
    im = img.transpose(2, 0, 1)
    im = img.reshape(img.shape[-1], -1)
    im_min = np.min(im, axis=1)
    im_max = np.max(im, axis=1)

    scale = 255 / (im_max - im_min)
    im_arr = (img - im_min) * scale

    return im_arr.astype(np.uint8)


def passed_arguments():
    parser = argparse.ArgumentParser(description= \
                                         "Script to run inference for segmentation models.")
    parser.add_argument("-d", "--data_path",
                        type=str,
                        required=True,
                        help="Path to directory containing datasets.")
    parser.add_argument("-c", "--config",
                        type=str,
                        required=True,
                        help="Path to .json model config file.")
    parser.add_argument("--data_map",
                        type=str,
                        default=None,
                        help="Path to .json file with train/val/test split for experiment.")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=True,
                        help="Optional path to model's checkpoint file. " + \
                             "If not specified, uses path from config.")
    parser.add_argument("--split",
                        nargs="+",
                        type=float,
                        default=[0.8, 0.1, 0.1],
                        help="Train/val/test split percentages.")
    parser.add_argument("-s", "--set_type",
                        type=str,
                        default="val",
                        help="One of the train/val/test sets to perform inference.")
    parser.add_argument("--classes",
                        type=str,
                        default=os.path.join("segmentation", "classes.json"),
                        help="Path to .json index->class name file.")
    parser.add_argument("--inf_out",
                        type=str,
                        default=None,
                        help="Path to output directory to store inference results. " + \
                             "Defaults to `.../data_path/inference/`")
    parser.add_argument('-n', '--name',
                        type=str,
                        default=None,
                        help='Experiment name, used as directory name in inf_dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = passed_arguments()

    set_type = args.set_type.lower()
    assert set_type in {"train", "val", "test"}, "Only train/val/test sets permitted."

    # Create config handler
    ch = ConfigHandler(args.data_path, args.config, args.classes, args.data_path,
                       args.inf_out, args.name)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model
    model = load_model(ch, from_checkpoint=args.checkpoint)
    model.to(device)
    model.eval()

    # Set up dataset
    dataset = create_dataset(
        ch.config,
        config_handler=ch,
        data_path=args.data_path,
        data_map_path=args.data_map,
        train_val_test=args.split,
        inf_mode=True
    )

    # Create dataset loaders for inference.
    b_size = ch.config.get("batch_size", 32)
    train_loader, val_loader, test_loader = dataset.create_data_loaders(batch_size=b_size)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[args.set_type]

    # Begin inference
    for batch_index, (input_t, y) in enumerate(loader):
        # Shift to correct device
        input_t, y = dataset.shift_sample_to_device((input_t, y), device)

        # Input into the model
        preds = model(input_t)

        # TODO: Fix inference for time series
        # Convert from tensor to numpy array
        # imgs = input_t.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        label_masks = y.detach().cpu().numpy()

        # TODO: Add back images to this loop later
        # Iterate over each image in batch.
        for ind, (pred, label_mask) in enumerate(zip(preds, label_masks)):
            _pred = pred[np.newaxis, ...]  # shape (b, #c, h, w)
            _label_mask = label_mask[np.newaxis, ...]  # shape (b, #c, h, w)

            _metrics = calculate_metrics(_pred, _label_mask, pred_threshold=0)

            # Find the count of each class in ground truth, record in metrics dict as whole num
            n = dataset.num_classes
            label_class_counts = np.count_nonzero(label_mask.reshape(n, -1), axis=-1)

            # Create metrics dict
            metrics_dict = create_metrics_dict(
                dataset.remapped_classes,
                iou=_metrics["iou"],
                prec=_metrics["prec"],
                recall=_metrics["recall"],
                gt_class_count=label_class_counts.tolist()
            )

            # Id for saving file.
            img_id = (batch_index * b_size) + ind

            # # Convert from b,g,r (indices/bands 3, 2, 1) to r,g,b
            # # Convert to (h, w, #c) shape and scale to uint8 values
            # img = img[1:4, ...][::-1]
            # img = img.transpose(1, 2, 0)
            # img = bytescale(img, high=255)
            #
            # # Save original image
            # im = Image.fromarray(img, mode="RGB")
            # im.save(os.path.join(ch.inf_dir, f"{img_id}_im.jpg"))
            #
            # # Draw pred mask on image
            # pred_im, pred_mask = draw_mask_on_im(img, pred > 0)
            # pred_im.save(os.path.join(ch.inf_dir, f"{img_id}_pred.jpg"))
            # pred_mask.save(os.path.join(ch.inf_dir, f"{img_id}_raw_pred.jpg"))
            #
            # # Draw ground truth mask on image
            # gt_im, gt_mask = draw_mask_on_im(img, label_mask)
            # gt_im.save(os.path.join(ch.inf_dir, f"{img_id}_gt.jpg"))
            # gt_mask.save(os.path.join(ch.inf_dir, f"{img_id}_raw_gt.jpg"))

            # Save eval results.
            with open(os.path.join(ch.inf_dir, f"{img_id}_metrics.json"), 'w') as f:
                json.dump(metrics_dict, f, indent=2)

    print(f"Inference complete for set {set_type}!")
