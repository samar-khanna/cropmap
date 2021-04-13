import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter
from models.model_utils import create_resnet_backbone


class FCN(nn.Module):
    def __init__(self, backbone, num_classes=1):
        super(FCN, self).__init__()

        # Get intermediate layer feature maps for upsampling.
        return_layers = {'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # Layer is a Sequential of either Bottlenecks (conv3) or BasicBlocks (conv2)
        layer = backbone.layer4
        last_conv = getattr(layer[-1], "conv3", layer[-1].conv2)
        out_channels = last_conv.out_channels
        self.classifier = FCNHead(in_channels=out_channels, channels=num_classes)

    @classmethod
    def create(cls, config, num_classes):
        """
        Creates a new FCN model given the config dictionary.
        Initialises a ResNet backbone (optionally pretrained) for the segmentation model.
        """
        backbone = create_resnet_backbone(config)

        # Create Segmentation model
        return cls(backbone, num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # contract: features is a dict of tensors
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


# Small test
if __name__ == "__main__":
    resnet18 = torchvision.models.resnet18()
    unet = FCN(resnet18)
    x = torch.randn(1, 3, 224, 224)
    out = unet(x)
    print(out.shape)
    assert out.shape == (1, 1, 224, 224)