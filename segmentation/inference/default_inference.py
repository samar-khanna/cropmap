import os
import json
import numpy as np
import torch.nn as nn
from PIL import Image
from typing import Optional

from utils.colors import get_cmap
from data_loaders.dataset import CropDataset
from inference.base_inference import InferenceAgent
from metrics import create_metrics_dict, confusion_matrix_from_images, mean_accuracy_from_images


class DefaultInferenceAgent(InferenceAgent):
    def __init__(
            self,
            model: nn.Module,
            dataset: CropDataset,
            batch_size: int,
            out_dir: str,
            exp_name: str,
            metric_names=(),
    ):
        """
        Handles inference, including saving metrics and TODO: images
        @param model: Segmentation model to evaluate
        @param dataset: CropDataset instance
        @param batch_size: Batch size of input images for inference
        @param out_dir: Output directory where inference results will be saved
        @param exp_name: Unique name for inference experiment
        @param metric_names: Names of metrics that will measure inference performance
        """
        super().__init__(model, dataset, batch_size, out_dir, exp_name, metric_names)

    def _format_for_save(self, im_arr):
        """
        Formats given pred/gt image for saving
        @param im_arr: (3, h, w) RGB image, with min val 0. and max val 1.
        @return: PIL RGB image object
        """
        # Format pred-mask to save
        im_arr = (im_arr * 255).astype(np.uint8)  # bytescaling
        im_arr = im_arr.transpose(1, 2, 0)  # (c, h, w) -> (h, w, c)
        im = Image.fromarray(im_arr, mode="RGB")
        return im

    def infer(self, set_type, save_images=False):
        """
        Runs inference for the model on the given set_type for the dataset.
        Saves metrics in inference out directory per ground truth mask.
        @param set_type: One of train/val/test
        @param save_images: Whether to store pred/gt image results in out dir
        """
        train_loaders, val_loaders, test_loaders = \
            self.dataset.create_data_loaders(batch_size=self.batch_size)
        loaders = {'train': train_loaders, 'val': val_loaders, 'test': test_loaders}
        data_loader = loaders[set_type]

        # Begin inference
        for batch_index, (input_t, y) in enumerate(data_loader):
            # Shift to correct device
            input_t, y = self.dataset.shift_sample_to_device((input_t, y), self.device)

            # Input into the model
            preds = self.model(input_t)

            # TODO: Fix inference for time series
            # Convert from tensor to numpy array
            # imgs = input_t.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()  # (b, #c, h, w)
            label_masks = y.detach().cpu().numpy()  # (b, #c, h, w)

            # TODO: Add back images to this loop later
            # Iterate over each image in batch.
            for ind, (pred, label_mask) in enumerate(zip(preds, label_masks)):
                _pred = pred[np.newaxis, ...]  # shape (b, #c, h, w)
                _label_mask = label_mask[np.newaxis, ...]  # shape (b, #c, h, w)

                # Get raw confusion matrix
                CM = confusion_matrix_from_images(_pred, _label_mask, pred_threshold=0)

                # TODO: How to do deal with this? Might have same issue as NaN
                # Get mean accuracy
                mean_acc = mean_accuracy_from_images(_pred, _label_mask, pred_threshold=0)

                # Find the count of each class in ground truth, record in metrics dict as whole num
                n = self.dataset.num_classes
                label_class_counts = np.count_nonzero(label_mask.reshape(n, -1), axis=-1)

                # Create metrics dict
                metrics_dict = create_metrics_dict(
                    self.dataset.remapped_classes,
                    accuracy=mean_acc,
                    tn=CM[:, 0, 0],
                    fp=CM[:, 0, 1],
                    fn=CM[:, 1, 0],
                    tp=CM[:, 1, 1],
                    gt_class_count=label_class_counts.tolist()
                )

                # Id for saving file.
                img_id = (batch_index * self.batch_size) + ind

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

                if save_images:
                    display_pred_gt = self.dataset.format_for_display(pred, label_mask)
                    pred_mask, gt_mask = display_pred_gt[0], display_pred_gt[1]

                    # Format pred-mask to save
                    pred_im = self._format_for_save(pred_mask)
                    pred_im.save(os.path.join(self.out_dir, f"{img_id}_raw_pred.png"))

                    gt_im = self._format_for_save(gt_mask)
                    gt_im.save(os.path.join(self.out_dir, f"{img_id}_raw_gt.png"))

                # Save eval results.
                with open(os.path.join(self.out_dir, f"{img_id}_metrics.json"), 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
