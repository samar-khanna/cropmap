import torch
import torch.nn as nn
from models.utils import NConvBlock

class DumbNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_conv=4, intermediate_channels=128):
        """
        Creates a simple segmentation network without any downsampling.
        """
        super().__init__()
        self.conv_layers = NConvBlock(in_channels, intermediate_channels, n=num_conv,
                                        kernel_size=1, padding=0)

        self.final_conv = nn.Conv2d(intermediate_channels, num_classes, kernel_size=1)

    @classmethod
    def create(cls, config, num_classes):
        """
        Creates SimpleNet given model config and number of classes.
        """
        in_channels = config.get("input_shape", [3])[0]
        return cls(in_channels, num_classes, **config["classifier_kwargs"])

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # dim is (b, block_length*c, h, w)
        x = self.conv_layers(x)
        out = self.final_conv(x)
        return out
