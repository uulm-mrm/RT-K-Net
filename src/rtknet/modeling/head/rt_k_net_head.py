import logging
from typing import Dict

import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.utils.registry import locate
from rtknet.layers import get_norm, group_feature_assembling
from rtknet.modeling.feature_map_generator import build_feature_map_generator
from torch import nn

from .kernel_update_head import PanopticKernelUpdateHead

__all__ = ["RTKNetHead"]


@SEM_SEG_HEADS_REGISTRY.register()
class RTKNetHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        feature_map_generator,
        in_channels,
        kernel_feature_channels,
        mask_feature_channels,
        num_kernels,
        num_classes,
        num_kernel_update_heads,
        normalize_sigmoid_masks,
        norm,
        init_func,
    ):
        super().__init__()

        self.feature_map_generator = feature_map_generator
        self.kernel_feature_channels = kernel_feature_channels
        self.num_kernels = num_kernels
        self.num_classes = num_classes
        self.num_kernel_update_heads = num_kernel_update_heads
        self.normalize_sigmoid_masks = normalize_sigmoid_masks

        self.feat_transform = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, kernel_feature_channels, kernel_size=(1, 1), bias=not norm),
            get_norm(norm, kernel_feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_feature_channels, kernel_feature_channels, kernel_size=(1, 1)),
            get_norm(norm, kernel_feature_channels),
        )
        self.init_kernels = nn.Conv2d(
            kernel_feature_channels,
            num_kernels,
            kernel_size=(1, 1),
            padding=0,
            bias=False,
        )

        # Auxiliary head for semantic segmentation loss. Can be dropped during inference
        self.sem_seg_head = nn.Sequential(
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, kernel_feature_channels, kernel_size=(1, 1), bias=not norm),
            get_norm(norm, kernel_feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                kernel_feature_channels,
                num_classes,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )

        heads = []
        for _ in range(self.num_kernel_update_heads):
            heads.append(
                PanopticKernelUpdateHead(
                    in_channels=kernel_feature_channels,
                    out_channels=mask_feature_channels,
                    num_classes=num_classes,
                    normalize_sigmoid_masks=normalize_sigmoid_masks,
                )
            )
        self.heads = nn.ModuleList(heads)

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
        feature_map_generator = build_feature_map_generator(cfg, input_shape)
        return {
            "feature_map_generator": feature_map_generator,
            "in_channels": cfg.MODEL.FEATURE_MAP_GENERATOR.OUT_CHANNELS,
            "kernel_feature_channels": cfg.MODEL.KERNEL_UPDATE_HEADS.KERNEL_FEATURE_CHANNELS,
            "mask_feature_channels": cfg.MODEL.KERNEL_UPDATE_HEADS.OUT_CHANNELS,
            "num_kernels": cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "num_kernel_update_heads": cfg.MODEL.SEM_SEG_HEAD.NUM_KERNEL_UPDATE_HEADS,
            "normalize_sigmoid_masks": cfg.MODEL.SEM_SEG_HEAD.NORMALIZE_SIGMOID_MASKS,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "init_func": cfg.MODEL.SEM_SEG_HEAD.INIT_FUNC,
        }

    def forward(self, features):
        feature_map = self.feature_map_generator(features)

        if self.training:
            # Generate aux semantic output
            seg_mask_pred = self.sem_seg_head(feature_map)

        feature_map = self.feat_transform(feature_map)
        mask_pred = self.init_kernels(feature_map)

        # Generate kernels
        batch_size = feature_map.shape[0]
        kernels = self.init_kernels.weight.clone()
        kernels = kernels[None].expand(batch_size, *kernels.size())
        group_features = group_feature_assembling(
            mask_pred, feature_map, self.normalize_sigmoid_masks
        )
        kernels = kernels + group_features.view(
            batch_size, self.num_kernels, self.kernel_feature_channels, 1, 1
        )

        cls_score = None
        if self.training:
            predictions_class = [cls_score]
            predictions_mask = [
                F.interpolate(mask_pred, scale_factor=2, mode="bilinear", align_corners=False)
            ]
        # Iterate through KernelUpdateHeads
        for stage in range(self.num_kernel_update_heads):
            refined_kernels, refined_mask_pred, cls_score = self.heads[stage](
                feature_map, kernels, mask_pred
            )
            kernels = refined_kernels
            mask_pred = refined_mask_pred

            # Add intermediate results for loss calculation
            if self.training:
                predictions_class.append(cls_score)
                predictions_mask.append(
                    F.interpolate(mask_pred, scale_factor=2, mode="bilinear", align_corners=False)
                )

        if self.training:
            return {
                "outputs": self._get_outputs_for_loss_calc(predictions_class, predictions_mask),
                "seg_mask_pred": F.interpolate(
                    seg_mask_pred, scale_factor=2, mode="bilinear", align_corners=False
                ),
                "feature_map": F.interpolate(
                    feature_map, scale_factor=2, mode="bilinear", align_corners=False
                ),
            }

        return {"pred_logits": cls_score, "pred_masks": mask_pred}

    @torch.jit.unused
    def _get_outputs_for_loss_calc(self, outputs_class, outputs_seg_masks):
        return [
            {"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class, outputs_seg_masks)
        ]
