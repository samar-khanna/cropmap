import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.resnet import BasicBlock
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter


class UpSample(nn.Module):
  """
  Inputs: x_up, x_down
  Performs: upsample (factor 2) |-> concat x_down, result |-> 2Conv(result)

  1. Upsample x_up by a factor of 2 (#channels = out_channels)
  2. Concatenate (prepend) x_down to result (#channels = 2x out_channels)
  3. Perform (Conv, BatchNorm, ReLU) x 2 on result (#channels = out_channels)
  """
  def __init__(self, in_channels, out_channels, use_bilinear=False):
    super(UpSample, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    
    self.double_conv = nn.Sequential(
      nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
    )
    if use_bilinear:
      self.up = nn.Sequential(
        nn.UpSample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.up = nn.ConvTranspose2d(in_channels, out_channels, 
                                   kernel_size=2, stride=2)

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


class UNet(nn.Module):
  """
  Model that creates a segmentation U-Net architecture.
  Modified to use ResNet backbones as the downsampling feature extractors.
  """
  def __init__(self, backbone, num_classes=1,
               use_bilinear=False):
    super(UNet, self).__init__()

    # Get intermediate layer feature maps for upsampling.
    return_layers = {"layer4":"layer4", "layer3":"layer3",
                     "layer2":"layer2", "layer1":"layer1",
                     "maxpool":"layer0"}
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
      up = UpSample(out_channels, in_channels, use_bilinear=use_bilinear)
      setattr(self, f"up{i}", up)
    
    # Final conv to reduce #channels to #classes
    self.final_conv = nn.Conv2d(self.up1.out_channels, num_classes, 
                                kernel_size=1, stride=1)


  def forward(self, x):
    input_shape = x.shape[-2:]

    down_sampled = self.backbone(x)
    
    x = down_sampled["layer4"]
    x = self.up4(x, down_sampled["layer3"])
    x = self.up3(x, down_sampled["layer2"])
    x = self.up2(x, down_sampled["layer1"])
    x = self.up1(x, down_sampled["layer0"])

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
