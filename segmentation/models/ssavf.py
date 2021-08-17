import torch
import torch.nn as nn
from libcpab import Cpab

from .model_utils import NConvBlock


class SSAVF(nn.Module):
    def __init__(self, num_classes, in_channels,
                 seq_len, n_conv, k_conv=1, dim_feature=64,
                 dim_theta_reg=16, tess=(6,), zero_boundary=True,
                 shift_method='random', theta_scale=0.5):
        super().__init__()
        self.dim_feature = dim_feature
        self.shift_method = shift_method
        self.theta_scale = theta_scale

        self.in_layer_norm = nn.LayerNorm(in_channels)
        if n_conv > 0:
            self.feature_extractor = NConvBlock(in_channels, dim_feature, conv_type='1d',
                                                n=n_conv, kernel_size=k_conv, use_bn=False,
                                                padding=0 if k_conv == 1 else 1)
            self.conv_layer_norm = nn.LayerNorm(dim_feature)
        else:
            self.feature_extractor = nn.Identity()
            self.conv_layer_norm = nn.Identity()

        # The velocity field class
        use_cuda = torch.cuda.is_available()  # TODO: Probably make this an init arg
        self.T = Cpab(tess, backend='pytorch', device='gpu' if use_cuda else 'cpu',
                      zero_boundary=zero_boundary, volume_perservation=False,)
        self.theta_d = self.T.get_theta_dim()

        # Regressor for the 3 * 2 affine matrix
        self.theta_reg = nn.Sequential(
            # TODO: Seq len needs to be known. Architecture dependent on seq len :(
            nn.Linear(2 * dim_feature * seq_len, 16),
            # nn.Conv1d(2 * dim_feature, dim_theta_reg, kernel_size=1),
            nn.ReLU(True),
            nn.Linear(dim_theta_reg, self.theta_d),
            nn.Tanh()  # Tanh constrains theta between -1 and 1
        )

        # Initialize the weights/bias with identity transformation
        self.theta_reg[-2].weight.data.zero_()
        self.theta_reg[-2].bias.data.copy_(
            torch.clone(self.T.identity(epsilon=0.001).view(-1))
        )


    @classmethod
    def create(cls, config, num_classes):
        in_channels = config.get("input_shape", [9])[0]

        return cls(num_classes, in_channels, **config["classifier_kwargs"])

    @staticmethod
    def images_to_pixels(x):
        t = len(x)
        b, c, h, w = x[0].shape

        x = torch.stack(x, dim=0)  # (t, b, in_c, h, w)
        x = x.permute(1, 3, 4, 2, 0).contiguous()  # (b, h, w, in_c, t)
        x = x.view(-1, c, t)  # (b*h*w, in_c, t)

        return x

    @staticmethod
    def pixels_to_images(x, h, w):
        # Call to contiguous to ensure subsequent viewing operations happen as expected
        N, c, t = x.shape
        x = x.view(-1, h, w, c, t)  # (b, h, w, in_c, t)
        x = x.permute(4, 0, 3, 1, 2).contiguous()  # (t, b, in_c, h, w)
        return [t for t in x]  # len t list, each tensor (b, in_c, h, w)

    def shift_data(self, x):
        N, in_c, t = x.shape  # (b*h*w, in_c, t)

        if 'random' in self.shift_method:
            theta = self.theta_scale * self.T.sample_transformation(N)  # (b*h*w, d)
        else:
            raise NotImplementedError

        out = self.T.transform_data(x, theta, outsize=(t, ))  # (b*h*w, in_c, t)

        return out, theta

    def localise(self, x):
        N, in_c, t = x.shape

        x = x.permute(0, 2, 1)  # (b*h*w, t, in_c)
        x = self.in_layer_norm(x)  # (b*h*w, t, in_c)

        x = x.permute(0, 2, 1)  # (b*h*w, in_c, t)
        x = self.feature_extractor(x)  # (b*h*w, c, t)

        x = x.permute(0, 2, 1)  # (b*h*w, t, c)
        x = self.conv_layer_norm(x)  # (b*h*w, t, c)

        x = x.permute(0, 2, 1)  # (b*h*w, c, t)
        return x

    def forward(self, x, x_shift=None):
        # x shape is (b*h*w, in_c, t)
        N, in_c, t = x.shape

        theta_shift = None
        if x_shift is None:
            x_shift, theta_shift = self.shift_data(x)

        x_orig_loc = self.localise(x)  # (b*h*w, c, t)
        x_shift_loc = self.localise(x_shift)  # (b*h*w, c, t)
        x_cat = torch.cat((x_orig_loc, x_shift_loc), dim=1)  # (b*h*w, 2*c, t)
        x_cat = x_cat.view(x_cat.shape[0], -1)  # (b*h*w, 2*c*t)

        theta_pred = self.theta_reg(x_cat)  # (b*h*w, d)

        # ASSUME: x and x_shift are already in shape (b*h*h, in_c, t)
        x_aligned = self.T.transform_data(x, theta_pred, outsize=(t, ))

        if theta_shift is None:
            return x_aligned, x_shift, theta_pred
        else:
            return x_aligned, x_shift, theta_pred, theta_shift
