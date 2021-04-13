import torch
from models.unet import UNet
from models.model_utils import create_resnet_backbone


class BlockUNet(UNet):
    """
    Model that creates UNet except uses concatenated input tensors (along channel dim)
    from time-series image data.
    """
    def __init__(self, *args, block_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_length = block_length

    @classmethod
    def create(cls, config, num_classes):
        """
        Creates a new BlockUNet model given the config dictionary.
        Initialises a ResNet backbone (optionally pretrained) for the segmentation model.
        """
        backbone = create_resnet_backbone(config)

        # Create Segmentation model
        return cls(backbone, num_classes=num_classes, **config["classifier_kwargs"])

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)  # dim is (b, block_length*c, h, w)
        return super().forward(x)
