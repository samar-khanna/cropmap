import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import NConvBlock


class Transformer(nn.Module):
    def __init__(self, num_classes, in_channels, dim_feature, num_layers,
                 dim_feedforward=256, nhead=4, dropout=0):
        """
        Creates a Transformer for images stacked in a time-sequence.
        Assumes all input time-shots will have same batch size.
        @param num_classes: Number of classes in prediction/label
        @param in_channels: Number of channels in each input image in time-series
        @param dim_feature: Encoding dimension (# channels) before inputted to transformer
        @param num_layers: Number of Transformer encoder layers in Transformer module
        @param dim_feedforward: Hidden number of nodes in fully connected net in Transformer
        @param nhead: Number of self attention heads
        @param dropout: Dropout fraction for the encoder layer
        """
        super().__init__()
        self.dim_feature = dim_feature
        self.pos_enc = PositionalEncoding(dim_feature)

        self.in_layer_norm = nn.LayerNorm(in_channels)
        self.first_conv = nn.Conv2d(in_channels, dim_feature, kernel_size=1)
        # self.conv_block = NConvBlock(in_channels, dim_feature, n=2, kernel_size=3)
        self.conv_layer_norm = nn.LayerNorm(dim_feature)

        encoder_layer = nn.TransformerEncoderLayer(
            dim_feature, nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.transformer_layer_norm = nn.LayerNorm(dim_feature)

        self.final_conv = nn.Conv2d(dim_feature, num_classes, kernel_size=1)

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

    def forward(self, x):
        # input is list of length t
        # each entry is (b, c, h, w)
        # concatenate to (t*b, c, h, w)
        # extract features
        # split back out into t x (b,c,h,w)
        # Transformer wants (t, b * h * w, c)

        # TODO: Do we need to worry about padded sequences with -1s?
        # TODO: Worry about 0s interacting with LayerNorm?
        t = len(x)
        b, c, h, w = x[0].shape

        x = torch.cat(x, dim=0)  # (t*b, in_c, h, w)
        x = x.permute(0, 2, 3, 1)  # (t*b, h, w, in_c)
        x = self.in_layer_norm(x)  # (t*b, h, w, in_c)

        x = x.permute(0, 3, 1, 2)  # (t*b, in_c, h, w)
        x = self.first_conv(x)  # (t*b, c, h, w)

        x = x.permute(0, 2, 3, 1)  # (t*b, h, w, c)
        x = self.conv_layer_norm(x)  # (t*b, h, w, c)

        # Call to contiguous to ensure the next viewing operation happens as expected
        x = x.permute(0, 3, 1, 2).contiguous()  # (t*b, c, h, w)
        x = x.view(t, -1, self.dim_feature)  # (t, bhw, c)
        x = self.pos_enc(x)  # (t, bhw, c)
        x = self.transformer_encoder(x)  # (t, bhw, c)
        x = self.transformer_layer_norm(x)  # (t, bhw, c)

        x = x.permute(1, 2, 0)  # (bhw, c, t)
        x = F.max_pool1d(x, kernel_size=x.shape[-1]).squeeze(-1)  # (bhw, c)

        x = x.view(b, self.dim_feature, h, w)  # (b, c, h, w)
        out = self.final_conv(x)  # (b, num_classes, h, w)
        return out


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
