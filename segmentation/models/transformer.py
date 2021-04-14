import torch
import torch.nn as nn
import math

# Feature extractor
#   num_conv, intermediate channels
#   id out the classifier layer
#   transformer encoder needs: torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
#   torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
#   d_model = intermediate_Channels (start w/ 64)
#   nhead can be 8 (default) for now, have in config
#   dim feedforward 256

class Transformer(nn.Module):
    def __init__(self, feature_extractor, num_classes, dim_feature, num_layers, dim_feedforward=256, nhead=4, dropout=0):
        """
        TODO
        """
        super().__init__()
        self.dim_feature = dim_feature
        self.feature_extractor = feature_extractor
        encoder_layer = nn.TransformerEncoderLayer(dim_feature, nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(dim_feature, num_classes)
        self.pos_enc = PositionalEncoding(dim_feature)

    @classmethod
    def create(cls, config, num_classes):
        """
        TODO: Creates SimpleNet given model config and number of classes.
        """
        from utils.loading import create_model
        feature_extractor = create_model(config['feature_extractor'], num_classes=1)  # create_model(model_config, num_classes)
        if config['feature_extractor']['classifier'] in ['SimpleNet', 'DumbNet']:
            feature_extractor.final_conv = nn.Identity()
        else:
            raise NotImplementedError

        return cls(feature_extractor, num_classes, **config["classifier_kwargs"])

    def forward(self, x):
        # input is list of length n
        # each entry is (b, c, h, w)
        # concatenate to (n*b, c, h, w)
        # extract features
        # split back out into n x (b,c,h,w)
        # Transformer wants (n, b * h * w, c)
        n = len(x)
        # print("Item shape:", x[0].shape)
        b, c, h, w = x[0].shape
        x = torch.cat(x, dim=0)
        # print("Catted shape", x.shape)
        x = self.feature_extractor(x)
        x = x.view(n, -1, self.dim_feature) # (n, bhw, c)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # print("Trans encoded shape", x.shape)
        final_features = torch.mean(x, dim=0) # (bhw, c)
        # print("Final feature shape", final_features.shape)
        out = self.linear(final_features) # (bhw, num_classes)
        out = out.view(b, out.shape[1], h, w)
        # print("Out shape", out.shape)
        #  asdf
        #print(self.linear.state_dict())
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
