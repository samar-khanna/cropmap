# This model is adapted from:
# Multi-stage Multi-recursive-input Fully Convolutional Networks for Neuronal Boundary Detection
# (Shen et. al.)

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from utils import NConvBlock, UpSampleAndMerge


class LightUpsample(nn.Module):
    """
    Inputs: x
    Performs: 1Conv (reduce channels) -> upsample (factor s) |-> 1Conv

    1. Perform (Conv, BatchNorm, ReLU) (#channels = mid_channels)
    2. Upsample result by factor s (#channels = mid_channels)
    3. Perform (Conv, BatchNorm, ReLU) (#channels = out_channels)
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, scale=2, use_bilinear=False):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.input_conv = NConvBlock(
            in_channels, mid_channels, n=1, kernel_size=1, padding=0
        )

        if use_bilinear:
            self.up = nn.Sequential(
                nn.UpSample(scale_factor=scale, mode='bilinear', align_corners=False),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(mid_channels)
            )
        else:
            # When scale > 2, then need scale-2 amount of padding to make sure out = in * scale
            self.up = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2,
                                         stride=scale, output_padding=scale-2)

        self.output_conv = NConvBlock(
            mid_channels, out_channels, n=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        return self.output_conv(self.up(self.input_conv(x)))


class M2UNet(nn.Module):
    """
    Model that simulates M2FCN except with a UNet-Resnet architecture
    Multi-Stage Multi-Recursive Input FCN (Shen et. al)

    Takes in sequence of inputs and processes in order. Different scale outputs from t-1 are
    upsampled to input image resolution and then concatenated with next input.
    """
    def __init__(self, backbone, num_classes=1, use_bilinear=False, recurrent_feature_depth=1):
        super().__init__()
        # Controls the #channels of features from time t-1 concatenated to input at time t
        self.r = recurrent_feature_depth

        # Get intermediate layer feature maps for upsampling.
        return_layers = {"layer4": "layer4", "layer3": "layer3",
                         "layer2": "layer2", "layer1": "layer1",
                         "maxpool": "layer0"}
        self.num_layers = len(return_layers)

        # Reformat first conv to take in new #channels
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(self.num_layers*self.r + old_conv.in_channels,
                                   old_conv.out_channels,
                                   kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride,
                                   padding=old_conv.padding)
        # backbone.conv1.weight[:, :3] = old_conv.weight

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # Calculate num channels and create an upsampling oparation
        scales = {1: 1, 2: 2, 3: 2, 4: 2}  # layer num --> upsample scale
        for l, s in scales.items():
            # Layer is a Sequential of either Bottlenecks (conv3) or BasicBlocks (conv2)
            layer = getattr(backbone, f"layer{l}")
            first_conv = layer[0].conv1
            last_conv = getattr(layer[-1], "conv3", layer[-1].conv2)
            in_channels = first_conv.in_channels
            out_channels = last_conv.out_channels

            # Flip order of (in_c, out_c) to (out_c, in_c) for upsampling
            up = UpSampleAndMerge(out_channels, in_channels, use_bilinear=use_bilinear, scale=s)
            setattr(self, f"up{l}", up)

            # Generate layer to take upsampled directly to input resolution
            direct_scale = s ** l if l > 1 else 4
            side_out = LightUpsample(in_channels, self.r, mid_channels=64, scale=direct_scale)
            setattr(self, f"side_out{l}", side_out)

        # Final conv to reduce #channels to #classes
        self.final_conv = nn.Conv2d(self.up1.out_channels, num_classes,
                                    kernel_size=1, stride=1)
        self.final_side_out = nn.Conv2d(num_classes, self.r, kernel_size=1, stride=1)

    def forward(self, x_list):
        x0 = x_list[0]
        b, c, h, w = x0.shape
        side_out = torch.zeros((b, self.num_layers, h, w), dtype=x0.dtype, device=x0.device)

        for i, x_in in enumerate(x_list):
            input_shape = x_in.shape[-2:]

            x_in = torch.cat((x_in, side_out), dim=1)
            down_sampled = self.backbone(x_in)

            x4 = down_sampled["layer4"]  # 1/32
            x3 = self.up4(x4, down_sampled["layer3"])  # 1/16
            x2 = self.up3(x3, down_sampled["layer2"])  # 1/8
            x1 = self.up2(x2, down_sampled["layer1"])  # 1/4
            x0 = self.up1(x1, down_sampled["layer0"])  # 1/4

            x_out = self.final_conv(x0)  # 1/4
            x_out = F.interpolate(x_out, size=input_shape, mode="bilinear", align_corners=False)

            # Don't need side outputs at final layer
            if i < len(x_list) - 1:
                # Get the side outputs and concatenate them along channels
                s4 = self.side_out4(x3)
                s3 = self.side_out3(x2)
                s2 = self.side_out2(x1)
                s1 = self.side_out1(x0)
                s0 = self.final_side_out(x_out)
                side_out = torch.cat((s0, s1, s2, s3, s4), dim=1)

        # Return upsampled out from final layer
        return x_out


if __name__ == "__main__":
    resnet = torchvision.models.resnet50()
    seq_unet = M2UNet(resnet)
    x_list = [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)]
    out = seq_unet(x_list)
    print(out.shape)
    assert out.shape == (1, 1, 224, 224)
