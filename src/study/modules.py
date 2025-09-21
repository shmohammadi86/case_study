"""
Neural network building blocks for autoencoder components.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block with BatchNorm and ReLU.

    Used as a building block in autoencoder architectures.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size for convolution
    stride : int
        Stride for convolution
    padding : int
        Padding for convolution
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out: torch.Tensor = self.conv(x)
        bn_out: torch.Tensor = self.bn(conv_out)
        output: torch.Tensor = self.relu(bn_out)
        return output


class DeconvBlock(nn.Module):
    """
    Deconvolutional block with BatchNorm and ReLU.

    Used for upsampling in decoder architectures.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size for transposed convolution
    stride : int
        Stride for transposed convolution
    padding : int
        Padding for transposed convolution
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deconv_out: torch.Tensor = self.deconv(x)
        bn_out: torch.Tensor = self.bn(deconv_out)
        output: torch.Tensor = self.relu(bn_out)
        return output
