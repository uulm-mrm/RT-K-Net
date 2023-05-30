import logging
from typing import Dict

import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import locate
from rtknet.layers import DeepAggregationPPM, get_norm
from torch import nn

from .build import FEATURE_MAP_GENERATOR_REGISTRY


@FEATURE_MAP_GENERATOR_REGISTRY.register()
class RTFormerHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        base_channels,
        out_channels,
        da_ppm_channels=128,
        head_channels=256,
        norm="BN",
        init_func=None,
    ):
        super().__init__()
        self.da_ppm = DeepAggregationPPM(
            base_channels * 8, da_ppm_channels, base_channels * 2, norm=norm
        )

        self.conv = nn.Sequential(
            get_norm(norm, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, head_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm, head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, out_channels, kernel_size=1, padding=0, bias=True),
        )

        self._init_func = locate(init_func) if init_func is not None else None
        self._logger = logging.getLogger(__name__)
        self._init_weights()

    def _init_weights(self):
        if self._init_func is None:
            return
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self._init_func(module)
                self._logger.info(
                    f"{name}: Init with "
                    f"{getattr(self._init_func, '__name__', repr(self._init_func))}"
                )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "base_channels": cfg.MODEL.RT_FORMER_HEAD.BASE_CHANNELS,
            "out_channels": cfg.MODEL.FEATURE_MAP_GENERATOR.OUT_CHANNELS,
            "head_channels": cfg.MODEL.RT_FORMER_HEAD.HEAD_CHANNELS,
            "norm": cfg.MODEL.RT_FORMER_HEAD.NORM,
            "init_func": cfg.MODEL.FEATURE_MAP_GENERATOR.INIT_FUNC,
        }

    def forward(self, features):
        x6 = self.da_ppm(features["rt5_low"])
        x6 = F.interpolate(x6, size=features["rt5_high"].shape[2:], mode="bilinear")
        x_out = torch.cat([features["rt5_high"], x6], dim=1)

        return self.conv(x_out)
