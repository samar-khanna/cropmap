import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.resnet import BasicBlock
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter
from models.utils import UpSampleAndMerge


class UNet(nn.Module):
    """
    Model that creates a segmentation U-Net architecture.
    Modified to use ResNet backbones as the downsampling feature extractors.
    """
    def __init__(self, backbone, num_classes=1,
                 use_bilinear=False):
        super().__init__()

        # Get intermediate layer feature maps for upsampling.
        return_layers = {"layer4": "layer4", "layer3": "layer3",
                         "layer2": "layer2", "layer1": "layer1",
                         "maxpool": "layer0"}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # Calculate num channels and create an upsampling oparation
        for i in range(4, 0, -1):
            # Layer is a Sequential of either Bottlenecks (conv3) or BasicBlocks (conv2)
            layer = getattr(backbone, f"layer{i}")
            first_conv = layer[0].conv1
            last_conv = getattr(layer[-1], "conv3", layer[-1].conv2)
            in_channels = first_conv.in_channels
            out_channels = last_conv.out_channels

            # Flip order of (in_c, out_c) to (out_c, in_c) for upsampling
            up = UpSampleAndMerge(out_channels, in_channels, use_bilinear=use_bilinear)
            setattr(self, f"up{i}", up)

        # Final conv to reduce #channels to #classes
        self.final_conv = nn.Conv2d(self.up1.out_channels, num_classes,
                                    kernel_size=1, stride=1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        down_sampled = self.backbone(x)

        x = down_sampled["layer4"]  # 1/32
        x = self.up4(x, down_sampled["layer3"])  # 1/16
        x = self.up3(x, down_sampled["layer2"])  # 1/8
        x = self.up2(x, down_sampled["layer1"])  # 1/4
        x = self.up1(x, down_sampled["layer0"])  # 1/4

        x = self.final_conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x


# Small test
if __name__ == "__main__":
    resnet = torchvision.models.resnet50()
    unet = UNet(resnet)
    x = torch.randn(1, 3, 224, 224)
    out = unet(x)
    print(out.shape)
    assert out.shape == (1, 1, 224, 224)
