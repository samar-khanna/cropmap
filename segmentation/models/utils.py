import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models.resnet as resnet


class NConvBlock(nn.Module):
    """
    Applies (Conv, BatchNorm, ReLU) x N on input
    """
    def __init__(self, in_channels, out_channels, n=2, kernel_size=3, padding=1):
        super().__init__()
        assert n > 0, "Need at least 1 conv block"

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        for _ in range(n - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class UpSampleAndMerge(nn.Module):
    """
    Inputs: x_up, x_down
    Performs: upsample (factor s) |-> concat x_down, result |-> 2Conv(result)

    1. Upsample x_up by a factor of s (#channels = out_channels)
    2. Concatenate (prepend) x_down to result (#channels = 2x out_channels)
    3. Perform (Conv, BatchNorm, ReLU) x 2 on result (#channels = out_channels)
    """
    def __init__(self, in_channels, out_channels, use_bilinear=False, scale=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_bilinear:
            self.up = nn.Sequential(
                nn.UpSample(scale_factor=scale, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2,
                stride=scale, output_padding=max(scale-2, 0)
            )

        # Since input to conv is concatenated tensor, we need 2*out_channels
        self.double_conv = NConvBlock(out_channels * 2, out_channels, n=2,
                                      kernel_size=3, padding=1)

    def forward(self, x_up, x_down):
        x_up = self.up(x_up)

        # Find size difference to perform center-crop (x_down is larger)
        # Shapes are in N x C x H x W format
        diff_H = x_down.shape[2] - x_up.shape[2]
        diff_W = x_down.shape[3] - x_up.shape[3]
        x_up = F.pad(x_up, [diff_H//2, diff_H - diff_H//2,
                            diff_W//2, diff_W - diff_W//2],
                     value=-1.0)

        # Concatenate feature maps along #channels dimension
        x_out = torch.cat((x_down, x_up), dim=1)
        x_out = self.double_conv(x_out)

        return x_out


def create_resnet_backbone(config):
    """
    Creates a ResNet backbone using the parameters described in config.
    The model config must contain a valid resnet name available in torchvision.models.resnet
    Replaces the input channels with the correct number of channels for this task.
    """
    backbone_name = config["backbone"].lower()
    assert backbone_name.find("resnet") > -1, "Only resnet backbones supported"

    # Initialise ResNet backbone
    backbone = resnet.__dict__[backbone_name](
        # replace_stride_with_dilation=[False, True, True],
        **config.get("backbone_kwargs", {})
    )

    # Reformat first conv to take in new #channels
    in_channels = config.get("input_shape", [3])[0]
    old_conv = backbone.conv1
    backbone.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                               kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride,
                               padding=old_conv.padding)
    if config.get("backbone_kwargs", {}).get("pretrained"):
        backbone.conv1.weight[:, :3] = old_conv.weight

    return backbone
