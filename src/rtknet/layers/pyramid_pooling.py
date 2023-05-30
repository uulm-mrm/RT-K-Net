# Adapted from PaddleSeg, ported to torch
# https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/paddleseg/models/rtformer.py
import torch
import torch.nn.functional as F
from torch import nn

from .batch_norm import get_norm

__all__ = ["DeepAggregationPPM"]


class DeepAggregationPPM(nn.Module):
    """
    Deep Aggregation Pyramid Pooling Module as proposed in
    "Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes"
    (https://arxiv.org/abs/2101.06085)
    """

    def __init__(self, in_channels, inter_channels, out_channels, norm="BN"):
        super().__init__()

        self.scale0 = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )

        self.process1 = nn.Sequential(
            get_norm(norm, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            get_norm(norm, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            get_norm(norm, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            get_norm(norm, inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )

        self.compression = nn.Sequential(
            get_norm(norm, inter_channels * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels * 5, out_channels, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x_shape = x.shape[2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1(
                (F.interpolate(self.scale1(x), size=x_shape, mode="bilinear") + x_list[0])
            )
        )
        x_list.append(
            (
                self.process2(
                    (F.interpolate(self.scale2(x), size=x_shape, mode="bilinear") + x_list[1])
                )
            )
        )
        x_list.append(
            self.process3(
                (F.interpolate(self.scale3(x), size=x_shape, mode="bilinear") + x_list[2])
            )
        )
        x_list.append(
            self.process4(
                (F.interpolate(self.scale4(x), size=x_shape, mode="bilinear") + x_list[3])
            )
        )

        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out
