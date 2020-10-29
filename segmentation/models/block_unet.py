import torch
from models.unet import UNet


class BlockUNet(UNet):
    """
    Model that creates UNet except uses concatenated input tensors (along channel dim)
    from time-series image data.
    """
    def __init__(self, *args, block_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_length = block_length

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)  # dim is (b, block_length*c, h, w)
        return super().forward(x)
