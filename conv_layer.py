import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(ConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass"""
        # x shape: [128, 1, 28, 28]
        # out_conv0 shape: [128, 256, 20, 20]
        out_conv0 = self.conv0(x)
        # out_relu shape: [128, 256, 20, 20]
        out_relu = self.relu(out_conv0)
        return out_relu
