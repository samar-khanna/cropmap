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
            self.feature_extractor = NConvBlock(in_channels, dim_feature, n_conv, k_conv,
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

    def images_to_pixels(self, x):
        t = len(x)
        b, c, h, w = x[0].shape

        x = torch.stack(x, dim=0)  # (t, b, in_c, h, w)
        x = x.permute(1, 3, 4, 2, 0).contiguous()  # (b, h, w, in_c, t)
        x = x.view(-1, c, t)  # (b*h*w, in_c, t)

        return x

    def shift_data(self, x):
        t = len(x)
        b, c, h, w = x[0].shape

        x = self.images_to_pixels(x)  # (b*h*w, in_c, t)

        if 'random' in self.shift_method:
            theta = self.theta_scale * self.T.sample_transformation(b*h*w)  # (b*h*w, d)
        else:
            raise NotImplementedError

        out = self.T.transform_data(x, theta, outsize=(t, ))  # (b*h*w, in_c, t)

        # Call to contiguous to ensure the next viewing operation happens as expected
        out = out.view(b, h, w, c, t)  # (b, h, w, in_c, t)
        out = out.permute(4, 0, 3, 1, 2).contiguous()  # (t, b, in_c, h, w)
        return [x for x in out], theta

    def localise(self, x):
        t = len(x)
        b, c, h, w = x[0].shape

        x = torch.cat(x, dim=0)  # (t*b, in_c, h, w)
        x = x.permute(0, 2, 3, 1)  # (t*b, h, w, in_c)
        x = self.in_layer_norm(x)  # (t*b, h, w, in_c)

        x = x.permute(0, 3, 1, 2)  # (t*b, in_c, h, w)
        # x = self.first_conv(x)  # (t*b, c, h, w)
        x = self.feature_extractor(x)  # (t*b, c, h, w)

        x = x.permute(0, 2, 3, 1)  # (t*b, h, w, c)
        x = self.conv_layer_norm(x)  # (t*b, h, w, c)

        x = x.view(t, b, h, w, self.dim_feature)  # (t, b, h, w, c)
        x = x.permute(1, 2, 3, 4, 0).contiguous()  # (b, h, w, c, t)
        x = x.view(-1, self.dim_feature, t)  # (b*h*w, c, t)

        return x

    def forward(self, x, x_shift=None):
        t = len(x)

        theta_shift = None
        if x_shift is None:
            x_shift, theta_shift = self.shift_data(x)

        x_orig_loc = self.localise(x)  # (b*h*w, c, t)
        x_shift_loc = self.localise(x_shift)  # (b*h*w, c, t)
        x_cat = torch.cat((x_orig_loc, x_shift_loc), dim=1)  # (b*h*w, 2*c, t)
        x_cat = x_cat.view(x_cat.shape[0], -1)  # (b*h*w, 2*c*t)

        theta_pred = self.theta_reg(x_cat)  # (b*h*w, d)

        x_pixels = self.images_to_pixels(x)  # (b*h*w, in_c, t)
        x_shift = self.images_to_pixels(x_shift)  # (b*h*w, in_c, t)
        x_aligned = self.T.transform_data(x_pixels, theta_pred, outsize=(t, ))

        if theta_shift is None:
            return x_aligned, x_shift, theta_pred
        else:
            return x_aligned, x_shift, theta_pred, theta_shift
