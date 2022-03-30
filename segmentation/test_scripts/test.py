import torch
import torch.nn
from torchvision.models.resnet import resnet18

from models.runet import SubUNet, RUNet
from models.m2unet import M2UNet
from models.model_utils import create_resnet_backbone


def runet_tests():
    # basic test
    x0 = torch.randn(2, 3, 128, 128)
    x1 = torch.randn(2, 3, 128, 128)
    x2 = torch.cat([torch.randn(1, 3, 128, 128), -1 * torch.ones(1, 3, 128, 128)])

    runet = RUNet(in_channels=3, num_classes=4)
    out = runet([x0, x1, x2])
    loss = (torch.ones(2, 4, 128, 128) - out).sum() ** 2
    loss.backward()
    assert out.shape == (2, 4, 128, 128)


def m2unet_tests():
    # basic test
    x0 = torch.randn(2, 3, 128, 128)
    x1 = torch.randn(2, 3, 128, 128)
    x2 = torch.cat([torch.randn(1, 3, 128, 128), -1 * torch.ones(1, 3, 128, 128)])

    backbone = create_resnet_backbone(
        {
            "backbone": "resnet34",
            "backbone_kwargs": {"pretrained": False},
            "input_shape": [3, 224, 224]
        }
    )
    m2unet = M2UNet(backbone, num_classes=4, use_bilinear=True)
    out = m2unet([x0, x1, x2])
    loss = (torch.ones(2, 4, 128, 128) - out).sum() ** 2
    loss.backward()
    assert out.shape == (2, 4, 128, 128)


if __name__ == "__main__":
    runet_tests()
    m2unet_tests()
