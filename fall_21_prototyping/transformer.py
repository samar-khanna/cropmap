import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, num_classes, in_channels, n_conv, k_conv=1, dim_feature=64,
                 num_layers=2, dim_feedforward=256, nhead=2, dropout=0):
        """
        Creates a Transformer for images stacked in a time-sequence.
        Assumes all input time-shots will have same batch size.
        @param num_classes: Number of classes in prediction/label
        @param in_channels: Number of channels in each input image in time-series
        @param n_conv: Number of conv blocks to extract conv features from each input
        @param k_conv: Kernel size for conv block to extract conv features.
        @param dim_feature: Encoding dimension (# channels) before inputted to transformer
        @param num_layers: Number of Transformer encoder layers in Transformer module
        @param dim_feedforward: Hidden number of nodes in fully connected net in Transformer
        @param nhead: Number of self attention heads
        @param dropout: Dropout fraction for the encoder layer
        """
        super().__init__()
        self.in_c = in_channels
        self.dim_feature = dim_feature
        self.pos_enc = PositionalEncoding(dim_feature)

        self.in_layer_norm = nn.LayerNorm(in_channels)
        self.feature_extractor = NConvBlock(in_channels, dim_feature, conv_type='1d',
                                            n=n_conv, kernel_size=k_conv, use_bn=False,
                                            padding=0 if k_conv == 1 else 1)
        # self.first_conv = nn.Conv2d(in_channels, dim_feature, kernel_size=1)
        self.conv_layer_norm = nn.LayerNorm(dim_feature)

        encoder_layer = nn.TransformerEncoderLayer(
            dim_feature, nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.transformer_layer_norm = nn.LayerNorm(dim_feature)

        self.final_conv = nn.Conv1d(dim_feature, num_classes, kernel_size=1)

    @classmethod
    def create(cls, config, num_classes):
        """
        Creates a Transformer from a config file, along with target number of classes.
        @param config: Dictionary config for the Transformer (should contain classifier_kwargs)
        @param num_classes: Number of classes in prediction/label
        @return: Instantiated Transformer
        """
        in_channels = config.get("input_shape", [9])[0]
        return cls(num_classes, in_channels, **config["classifier_kwargs"])

    def forward(self, x, return_final_feature=False):
        # TODO: Do we need to worry about padded sequences with -1s?
        # TODO: Worry about 0s interacting with LayerNorm?
        N, time_channel = x.shape
        x = x.view(N, -1, self.in_c)  # (N, t, in_c)
        t = x.shape[1]

        x = self.in_layer_norm(x)  # (N, t, in_c)

        x = x.permute(0, 2, 1)  # (N, in_c, t)
        x = self.feature_extractor(x)  # (N, c, t)

        x = x.permute(0, 2, 1)  # (N, t, c)
        x = self.conv_layer_norm(x)  # (N, t, c)

        x = x.permute(1, 0, 2)  # (t, N, c)
        x = self.pos_enc(x)  # (t, N, c)
        x = self.transformer_encoder(x)  # (t, N, c)
        x = self.transformer_layer_norm(x)  # (t, N, c)

        x = x.permute(1, 2, 0)  # (N, c, t)
        x = F.max_pool1d(x, kernel_size=x.shape[-1])  # (N, c, 1)

        out = self.final_conv(x)  # (N, num_classes, 1)
        out = out.squeeze(-1)
        if return_final_feature:
            return out, x.squeeze()
        else:
            return out  # (N, num_classes)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NConvBlock(nn.Module):
    """
    Applies (Conv, BatchNorm, ReLU) x N on input
    """
    def __init__(self, in_channels, out_channels, n=2,
                 conv_type='2d', kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        assert n > 0, "Need at least 1 conv block"
        assert conv_type in {'1d', '2d', '3d'}, "Only 1d/2d/3d convs accepted"

        if conv_type.lower() == '1d':
            layer = nn.Conv1d
        elif conv_type.lower() == '2d':
            layer = nn.Conv2d
        elif conv_type.lower() == '3d':
            layer = nn.Conv3d
        else:
            raise NotImplementedError

        layers = [layer(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())

        for _ in range(n - 1):
            layers.append(
                layer(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)
