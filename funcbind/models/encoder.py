import numpy as np
import torch
from torch import nn

from funcbind.models.unet3d import BlockUncond, MPConv, resample


class MPEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        level_channels=[32, 64, 128],
        resample_mode="keep",
        n_downsampling=0,
        downsample_first_block=True,
    ):
        """3D CNN encoder
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            level_channels (list, optional): the number of channels at each level (count top-down).
                Defaults to [32, 64, 128].
            resample_mode (str, optional): the mode of resampling.
                Defaults to "keep".
            n_downsampling (int, optional): the number of downsampling layers.
                Defaults to 0.
            downsample_first_block (bool, optional): whether to downsample the first block.
                Defaults to True.
        """
        super(MPEncoder, self).__init__()
        self.enc_blocks = nn.ModuleList()
        n_downsampling_counter = n_downsampling
        for i in range(len(level_channels)):
            in_ch = in_channels if i == 0 else level_channels[i - 1]
            out_ch = level_channels[i]
            if i == 0 and not downsample_first_block:
                self.enc_blocks.append(BlockUncond(in_ch, out_ch, flavor='enc', resample_mode="keep"))
            else:
                self.enc_blocks.append(
                    BlockUncond(in_ch, out_ch, flavor='enc', resample_mode=resample_mode if n_downsampling_counter >= 1 else "keep")
                )
                n_downsampling_counter -= 1
        self.fc = MPConv(out_ch, out_channels, kernel=[1,1,1])

    def forward(self, voxels):
        out = voxels
        for block in self.enc_blocks:
            out = block(out)
        out = self.fc(out)
        return out


def sample_posterior(moments, sample_posterior=True, save_log_var=True, deterministic=False):
    posterior = DiagonalGaussianDistribution(moments, save_log_var=save_log_var, deterministic=deterministic)
    if sample_posterior:
        z = posterior.sample()
    else:
        z = posterior.mode()

    return z, posterior


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, save_log_var=True):
        self.parameters = parameters
        self.mean, logvar = torch.chunk(parameters, 2, dim=1 if len(parameters.shape) == 5 else 2)
        self.deterministic = deterministic
        if not self.deterministic:
            logvar = torch.clamp(logvar, -30.0, 20.0)
            if save_log_var:
                self.logvar = logvar
            self.std = torch.exp(0.5 * logvar)
        else:
            del logvar

    def sample(self):
        if self.deterministic:
            x = self.mean
        else:
            x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            var = torch.exp(self.logvar)
            dim = [1, 2, 3, 4] if len(self.mean.shape) == 5 else [1, 2]
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2) + var - 1.0 - self.logvar, dim=dim)
            else:
                return 0.5 * torch.mean(torch.pow(self.mean - other.mean, 2) / other.var + var / other.var - 1.0 - self.logvar + other.logvar, dim=dim)

    def nll(self, sample):
        dim = [1, 2, 3, 4] if len(self.mean.shape) == 5 else [1, 2]
        if self.deterministic:
            return torch.Tensor([0.])
        var = torch.exp(self.logvar)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.mean(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / var, dim=dim)

    def mode(self):
        return self.mean


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        level_channels=[32, 64, 128],
        bottleneck_channel=1024,
        out_channel=1024,
        downsample_map=[False, False, False],
    ):
        """3D-Unet
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            level_channels (list, optional): the number of channels at each level (count top-down).
                Defaults to [32, 64, 128].
            bottleneck_channel (int, optional): the number of bottleneck channels.
                Defaults to 256.
        """
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList()
        for i in range(len(level_channels)):
            in_ch = in_channels if i == 0 else level_channels[i - 1]
            out_ch = level_channels[i]
            self.enc_blocks.append(Conv3DBlock(in_channels=in_ch, out_channels=out_ch, dilation=1))
        self.bottleNeck = Conv3DBlock(in_channels=out_ch, out_channels=bottleneck_channel)
        self.fc = nn.Conv3d(bottleneck_channel, out_channel, kernel_size=(1, 1, 1))

        self.downsample_map = downsample_map
        assert len(level_channels) == len(downsample_map)

    def forward(self, voxels):
        out = voxels
        for i, block in enumerate(self.enc_blocks):
            if self.downsample_map[i]:
                out = resample(out, mode="down")
            out = block(out)
        out = self.bottleNeck(out)
        out = self.fc(out)
        return out


class SingleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, non_linearity=True):
        super(SingleConv3D, self).__init__()
        self.non_linearity = non_linearity
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=dilation,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm3d(num_features=out_channels)

        if non_linearity:
            self.nl = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        if self.non_linearity:
            x = self.nl(x)
        return x


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        """The basic block for double 3x3x3 convolutions in the analysis path

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            bottleneck (bool, optional): specifies the bottlneck block. Defaults to False.
            use_bn (bool, optional): specifies use batchnorm layers or not.Default to True.
            res_block (bool, optional): use or not residual connection on conv block.Default to True.
        """
        super(Conv3DBlock, self).__init__()
        # first conv
        level_channels_in = [
            in_channels,
            out_channels,
            out_channels // 2,
        ]
        level_channels_out = [
            out_channels,
            out_channels // 2,
            out_channels,
        ]
        self.conv_layers = nn.ModuleList()
        for i in range(len(level_channels_in)):
            self.conv_layers.append(
                SingleConv3D(
                    in_channels=level_channels_in[i],
                    out_channels=level_channels_out[i],
                    non_linearity=(i != len(level_channels_out) - 1),
                    dilation=dilation,
                )
            )
        # non linearity
        self.nl = nn.ReLU()

    def forward(self, input):
        res = input
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(res)
                res = x.clone()
            else:
                res = conv(res)
        res += x
        res = self.nl(res)
        out = res
        return out
