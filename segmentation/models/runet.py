import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DownSample, UpSampleAndMerge, NConvBlock


class SubUNet(nn.Module):
    def __init__(self, in_channels=64, layer=3, use_maxpool=True, use_bilinear=True):
        """
        Sub UNet replacing the bottom-U portion of the network up to lth pooling layer.
        (i.e. output is upsampled to same resolution as output of lth pooling layer).
        Performs 4 - l + 1 downsampling operations, and 4 - l upsampling operations.
        @param in_channels: Number of channels of input into sub-network
        @param layer: layer index (1 to 4) indicating layer where this subnetwork is used
        @param use_maxpool: Whether to downsample with maxpool or conv
        @param use_bilinear: Whether to upsample with
        """
        assert 0 < layer <= 4
        assert in_channels % 2 == 0
        super().__init__()

        self.layer = layer

        self.down_samplers = nn.ModuleList()
        self.up_samplers = nn.ModuleList()

        # First downsample is independent of upsample
        self.down_samplers.append(
            DownSample(in_channels, in_channels * 2, use_maxpool=use_maxpool)
        )

        in_c = in_channels * 2
        for l in range(max(layer, 1), 4):
            self.down_samplers.append(
                DownSample(in_c, in_c*2, use_maxpool=use_maxpool)
            )
            self.up_samplers.append(
                UpSampleAndMerge(in_c*2, in_c, use_bilinear=use_bilinear)
            )
            in_c *= 2

    def forward(self, x):
        # Down sample input x by chaining down sampling layers
        down_sampled = [x]
        for down_sampler in self.down_samplers:
            x_down = down_sampler(down_sampled[-1])
            down_sampled.append(x_down)

        # Combine prev upsample output with corresponding down sampled input
        x_out = down_sampled[-1]
        for i, up_sampler in enumerate(reversed(self.up_samplers)):
            x_out = up_sampler(x_out, down_sampled[-(i+2)])  # starts at -2 to -num_layers

        return x_out


class SRU(nn.Module):
    def __init__(self, fz, fh, out_conv):
        """
        Implements the Single-Gated Recurrent Unit in the R-UNet paper
        @param fz: the backbone network, a Sub UNet
        @param fh: the gate network, a Sub UNet
        @param out_conv: Final convolution to produce output of SRU
        """
        super().__init__()
        self.fz = fz
        self.sigmoid = nn.Sigmoid()

        self.fh = fh
        self.tanh = nn.Tanh()

        self.out_conv = out_conv

    def forward(self, x, h_prev):
        z = self.sigmoid(self.fz(x))

        h_hat = self.tanh(self.fh(x))

        h_next = ((1 - z) * h_hat) + (z * h_prev)
        out = self.out_conv(h_next)

        return out, h_next


class RUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, layer: int = 3,
                 channel_list=(16, 16, 32, 64, 128), use_maxpool=True, use_bilinear=True):
        """
        Implements the R-UNet (recurrent UNet) as detailed: https://arxiv.org/pdf/1906.04913.pdf
        @param in_channels: Number of channels in input image
        @param num_classes: Number of classes in prediction/label
        @param layer: Layer (1 to 4) after which to replace portion with recurrent sub network
        @param channel_list: List of in channels (after input) for layers 1 to 4
        @param use_maxpool: Whether to downsample with 2x2 maxpool or 2x2 conv
        @param use_bilinear: Whether to upsample with bilinear or with conv transpose
        """
        super().__init__()

        self.num_classes = num_classes

        self.down_samplers = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        for l in range(0, layer):
            # First layer is just 2Conv, rest downsample with 2x2 Max Pool |-> 2Conv
            in_c, out_c = channel_list[l-1], channel_list[l]
            self.down_samplers.append(
                NConvBlock(in_channels + 1, channel_list[0], n=2) if l == 0 else
                DownSample(in_c, out_c, scale=2, n=2, use_maxpool=use_maxpool)
            )

            in_c, out_c = channel_list[l+1], channel_list[l]
            self.up_samplers.append(
                UpSampleAndMerge(in_c, out_c, scale=2, use_bilinear=use_bilinear)
            )

        # Hidden tensor is at lower scale than input to subnetwork
        self.hidden_c = channel_list[layer]  # for l=3, is 32
        self.hidden_scale = 2 ** layer  # for l=3, is 8
        self.sru = SRU(
            fz=SubUNet(channel_list[layer-1], layer, use_maxpool, use_bilinear),
            fh=SubUNet(channel_list[layer-1], layer, use_maxpool, use_bilinear),
            out_conv=NConvBlock(self.hidden_c, self.hidden_c, n=2)
        )

        # Final conv to reduce #channels to #classes
        self.final_conv = nn.Conv2d(channel_list[0], num_classes, kernel_size=1, stride=1)

    @classmethod
    def create(cls, config, num_classes):
        """
        Creates a new R-UNet given the model config and number of classes to train on.
        @param config: Dictionary config of R-UNet keyword arguments (except num_classes)
        @param num_classes: Number of classes to train on
        @return: An instantiated R-UNet model
        """
        in_c = config["input_shape"][0]
        return cls(in_channels=in_c, num_classes=num_classes, **config["classifier_kwargs"])

    def forward(self, x_list):
        # ASSUMPTION: x0 has max batch size with all valid samples.
        x0 = x_list[0]
        b, c, h, w = x0.shape
        device = x0.device

        # Get mask of valid valid samples for each batched element in time series input
        valid_in_batch = [~(x == -1).view(-1, c*h*w).all(dim=-1) for x in x_list]

        # Hidden tensor and output from previous time step
        hidden_size = (b, self.hidden_c, h//self.hidden_scale, w//self.hidden_scale)
        h_prev = torch.zeros(hidden_size, dtype=x0.dtype, device=device)
        prev_out = torch.zeros((b, 1, h, w), dtype=x0.dtype, device=device)

        # Final_out will be based on max possible batch size
        final_out = torch.zeros((b, self.num_classes, h, w), dtype=x0.dtype, device=device)
        for i, x_in in enumerate(x_list):
            input_shape = x_in.shape[-2:]
            x_in = torch.cat((x_in, prev_out), dim=1)  # shape (b, in_channels + 1, h, w)

            # Down sample until SRU/DRU
            is_valid_mask = valid_in_batch[i]
            down_sampled = [x_in[is_valid_mask]]
            for down_sampler in self.down_samplers:
                x_down = down_sampler(down_sampled[-1])
                down_sampled.append(x_down)

            # If l=3, down_sampled[-1] is shape: (b, channel_list[2], h/4, w/4)
            # If l=3, h_prev, h_next & x_out shape: (b, channel_list[3], h/8, w/8)
            # Only use SRU with hidden tensors of valid batch elements
            x_out, h_next = self.sru(down_sampled[-1], h_prev[is_valid_mask])

            # Up sample till (b, channel_list[0], h, w)
            for i, up_sampler in enumerate(reversed(self.up_samplers)):
                x_out = up_sampler(x_out, down_sampled[-(i + 1)])  # starts at -1 to 1

            x_final = self.final_conv(x_out)  # (b, num_classes, h, w)
            if x_final.shape[-2:] != input_shape:
                x_final = F.interpolate(
                    x_final, size=input_shape, mode="bilinear", align_corners=False
                )

            # Create (b, 1, h, w) map of predicted class per pixel
            x_pred = F.softmax(x_final, dim=1).argmax(dim=1, keepdim=True) + 1.

            # Update the out variables corresponding to valid elements in the batch
            h_prev[is_valid_mask] = h_next
            prev_out[is_valid_mask] = x_pred
            final_out[is_valid_mask] = x_final

        return final_out
