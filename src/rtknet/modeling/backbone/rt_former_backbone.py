# Adapted from PaddleSeg, ported to torch
# https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/paddleseg/models/rtformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from rtknet.layers import get_norm
from timm.models.layers import DropPath

__all__ = ["build_rt_former_backbone"]

PARAMS = {
    "slim": {"use_injection": [True, True], "base_channels": 32},
    "base": {"use_injection": [True, False], "base_channels": 64},
}


@BACKBONE_REGISTRY.register()
def build_rt_former_backbone(cfg, input_shape) -> nn.Module:
    variant_params = PARAMS[cfg.MODEL.RT_FORMER_BACKBONE.VARIANT]
    return RTFormerBackbone(norm=cfg.MODEL.RT_FORMER_BACKBONE.NORM, **variant_params)


class RTFormerBackbone(Backbone):
    def __init__(
        self,
        layer_nums=None,
        base_channels=64,
        num_heads=8,
        drop_rate=0.0,
        drop_path_rate=0.2,
        use_injection=None,
        cross_size=12,
        in_channels=3,
        norm="BN",
    ):
        super().__init__()
        if use_injection is None:
            use_injection = [True, False]
        if layer_nums is None:
            layer_nums = [2, 2, 2, 2]
        self.base_channels = base_channels
        base_chs = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            get_norm(norm, base_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            get_norm(norm, base_chs),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs, layer_nums[0], norm=norm)
        self.layer2 = self._make_layer(
            BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2, norm=norm
        )
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2, norm=norm
        )
        self.layer3_ = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2, 1, norm=norm)
        self.compression3 = nn.Sequential(
            get_norm(norm, base_chs * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_chs * 4, base_chs * 2, kernel_size=1, bias=False),
        )
        self.layer4 = EABlock(
            in_channels=[base_chs * 2, base_chs * 4],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[0],
            use_cross_kv=True,
            cross_size=cross_size,
            norm=norm,
        )
        self.layer5 = EABlock(
            in_channels=[base_chs * 2, base_chs * 8],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[1],
            use_cross_kv=True,
            cross_size=cross_size,
            norm=norm,
        )

        # Defines for self.output_shape()
        self._out_features = [
            "stem",
            "rt1",
            "rt2",
            "rt3_high",
            "rt3_low",
            "rt4_high",
            "rt4_low",
            "rt5_high",
            "rt5_low",
        ]
        self._out_feature_channels = {
            "stem": 4,
            "rt1": 4,
            "rt2": 8,
            "rt3_high": 8,
            "rt3_low": 16,
            "rt4_high": 8,
            "rt4_low": 32,
            "rt5_high": 8,
            "rt5_low": 32,
        }
        self._out_feature_strides = {
            "stem": base_chs,
            "rt1": base_chs,
            "rt2": base_chs * 2,
            "rt3_high": base_chs * 2,
            "rt3_low": base_chs * 4,
            "rt4_high": base_chs * 2,
            "rt4_low": base_chs * 8,
            "rt5_high": base_chs * 2,
            "rt5_low": base_chs * 8,
        }

    @staticmethod
    def _make_layer(block, in_channels, out_channels, blocks, stride=1, norm="BN"):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm(norm, out_channels),
            )

        layers = [block(in_channels, out_channels, stride, downsample, norm=norm)]
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(out_channels, out_channels, stride=1, no_relu=True, norm=norm))
            else:
                layers.append(block(out_channels, out_channels, stride=1, no_relu=False, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)  # c, 1/4
        outputs["stem"] = x
        x = self.layer1(x)  # c, 1/4
        outputs["rt1"] = x
        x = self.layer2(self.relu(x))  # 2c, 1/8
        outputs["rt2"] = x
        x3 = self.layer3(self.relu(x))  # 4c, 1/16
        x3_ = x + F.interpolate(
            self.compression3(x3), size=x.shape[2:], mode="bilinear"
        )  # 2c, 1/8  # noqa
        x3_ = self.layer3_(self.relu(x3_))  # 2c, 1/8
        outputs["rt3_high"] = x3_
        outputs["rt3_low"] = x3
        x4_, x4 = self.layer4([self.relu(x3_), self.relu(x3)])  # 2c, 1/8; 8c, 1/16
        outputs["rt4_high"] = x4_
        outputs["rt4_low"] = x4
        x5_, x5 = self.layer5([self.relu(x4_), self.relu(x4)])  # 2c, 1/8; 8c, 1/32
        outputs["rt5_high"] = x5_
        outputs["rt5_low"] = x5

        return outputs


class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, no_relu=False, norm="BN"
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = get_norm(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = get_norm(norm, out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out if self.no_relu else self.relu(out)


class EABlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=8,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_injection=True,
        use_cross_kv=True,
        cross_size=12,
        norm="BN",
    ):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        # low resolution
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                get_norm(norm, in_channels_l),
                nn.Conv2d(
                    in_channels_l, out_channels_l, kernel_size=1, stride=2, padding=0, bias=False
                ),
            )
        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=out_channels_l,
            num_heads=num_heads,
            use_cross_kv=False,
            norm=norm,
        )
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate, norm=norm)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # compression
        self.compression = nn.Sequential(
            get_norm(norm, out_channels_l),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_l, out_channels_h, kernel_size=1, bias=False),
        )

        # high resolution
        self.attn_h = ExternalAttention(
            in_channels_h,
            in_channels_h,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv,
            norm=norm,
        )
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate, norm=norm)
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                get_norm(norm, out_channels_l),
                nn.AdaptiveMaxPool2d(output_size=(self.cross_size, self.cross_size)),
                nn.Conv2d(
                    out_channels_l,
                    2 * out_channels_h,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )

        # injection
        if use_injection:
            self.down = nn.Sequential(
                get_norm(norm, out_channels_h),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels_h,
                    out_channels_l // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                get_norm(norm, out_channels_l // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels_l // 2,
                    out_channels_l,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )

    def forward(self, x):
        x_h, x_l = x

        # low resolution
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))  # n,out_chs_l,h,w

        # compression
        x_h_shape = x_h.shape[2:]
        x_l_cp = self.compression(x_l)
        x_h = x_h + F.interpolate(x_l_cp, size=x_h_shape, mode="bilinear")

        # high resolution
        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))  # n,out_chs_h,h,w
        else:
            cross_kv = self.cross_kv(x_l)  # n,2*out_channels_h,12,12
            cross_kv_split = torch.split(cross_kv, self.out_channels_h, dim=1)
            cross_k, cross_v = cross_kv_split[0], cross_kv_split[1]
            cross_k = cross_k.permute(0, 2, 3, 1).reshape(
                -1, self.out_channels_h, 1, 1
            )  # n*144,out_channels_h,1,1
            cross_v = cross_v.reshape(
                -1, self.cross_size * self.cross_size, 1, 1
            )  # n*out_channels_h,144,1,1
            x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k, cross_v))  # n,out_chs_h,h,w

        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        # injection
        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l


class ExternalAttention(nn.Module):
    def __init__(
        self, in_channels, out_channels, inter_channels, num_heads=8, use_cross_kv=False, norm="BN"
    ):
        super().__init__()
        assert (
            out_channels % num_heads == 0
        ), "out_channels ({}) should be be a multiple of num_heads ({})".format(
            out_channels, num_heads
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = get_norm(norm, in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert (
                self.same_in_out_chs
            ), "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.register_parameter(
                name="k", param=nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 1))
            )
            self.register_parameter(
                name="v", param=nn.Parameter(torch.zeros(out_channels, inter_channels, 1, 1))
            )
            torch.nn.init.normal_(self.k, std=0.001)
            torch.nn.init.normal_(self.v, std=0.001)

    def _act_sn(self, x):
        N, C, H, W = x.shape
        x = x.view([-1, self.inter_channels, H, W]) * (self.inter_channels**-0.5)
        x = F.softmax(x, dim=1)
        x = x.view([1, -1, H, W])
        return x

    def _act_dn(self, x):
        B, _, h, w = x.shape
        x = x.view([B, self.num_heads, self.inter_channels // self.num_heads, -1])
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.view([B, self.inter_channels, h, w])
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        """
        Args:
            x (Tensor): The input tensor.
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(
                x, self.k, bias=None, stride=2 if not self.same_in_out_chs else 1, padding=0
            )  # n,c_in,h,w -> n,c_inter,h,w
            x = self._act_dn(x)  # n,c_inter,h,w
            x = F.conv2d(x, self.v, bias=None, stride=1, padding=0)  # n,c_inter,h,w -> n,c_out,h,w
        else:
            assert (cross_k is not None) and (
                cross_v is not None
            ), "cross_k and cross_v should not be None when use_cross_kv"
            N, C, H, W = x.shape
            assert N > 0, f"The first dim of x ({N}) should be greater than 0"

            x = x.view(1, N * C, H, W)  # n,c_in,h,w -> 1,n*c_in,h,w
            x = F.conv2d(
                x, cross_k, bias=None, stride=1, padding=0, groups=N
            )  # 1,n*c_in,h,w -> 1,n*144,h,w  (group=B)
            x = self._act_sn(x)
            x = F.conv2d(
                x, cross_v, bias=None, stride=1, padding=0, groups=N
            )  # 1,n*144,h,w -> 1, n*c_in,h,w  (group=B)
            x = x.view([-1, self.in_channels, H, W])  # 1, n*c_in,h,w -> n,c_in,h,w  (c_in = c_out)
        return x


class MLP(nn.Module):
    def __init__(
        self, in_channels, hidden_channels=None, out_channels=None, drop_rate=0.0, norm="BN"
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = get_norm(norm, in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
