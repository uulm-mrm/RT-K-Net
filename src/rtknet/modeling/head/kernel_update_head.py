import logging

import torch
import torch.nn.functional as F
from detectron2.utils.registry import locate
from rtknet.layers import (
    AdaptiveKernelUpdate,
    KernelInteraction,
    bias_init_with_prob,
    get_norm,
    group_feature_assembling,
)
from torch import nn

__all__ = ["PanopticKernelUpdateHead"]


class PanopticKernelUpdateHead(nn.Module):
    """
    Panoptic Kernel Update head as proposed in "K-Net: Towards Unified Image Segmentation"
    (https://arxiv.org/abs/2106.14855)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        normalize_sigmoid_masks,
        num_heads=8,
        dropout=0.0,
        norm="LN",
        linear_init_func="rtknet.layers.weight_init.xavier_uniform_init_without_bias",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.normalize_sigmoid_masks = normalize_sigmoid_masks

        self.adaptive_kernel_update = AdaptiveKernelUpdate()
        self.kernel_interaction = KernelInteraction(
            in_channels=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            norm=norm,
        )

        self.fc_mask = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
        )
        self.fc_cls = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            get_norm(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_classes),
        )

        self._linear_init_func = locate(linear_init_func) if linear_init_func is not None else None
        self._logger = logging.getLogger(__name__)
        self._init_weights()

    def _init_weights(self):
        if self._linear_init_func is None:
            return
        # Initialize all Linear layers with the same init function
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                self._linear_init_func(module)
                self._logger.info(
                    f"{name}: Init with "
                    f"{getattr(self._linear_init_func, '__name__', repr(self._linear_init_func))}"
                )

        # Special initializations
        # We use focal loss, hence we need to initialize fc_cls according to bias_init_with_prob
        nn.init.constant_(self.fc_cls[-1].bias, bias_init_with_prob(0.01))
        self._logger.info("fc_cls.3.bias: Init with nn.init.constant_(bias_init_with_prob(0.01))")

    def forward(self, feature_map, kernels, mask_pred):
        B, N = kernels.shape[:2]
        C, H, W = feature_map.shape[-3:]

        # Dynamic kernel update
        # [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        kernels = kernels.reshape(B, N, self.in_channels, -1).permute(0, 1, 3, 2)
        group_features = group_feature_assembling(
            mask_pred, feature_map, self.normalize_sigmoid_masks
        )
        with torch.cuda.amp.autocast(enabled=self.normalize_sigmoid_masks):
            refined_kernels = self.adaptive_kernel_update(group_features, kernels)
        refined_kernels = self.kernel_interaction(refined_kernels, B, N)

        # Refine mask prediction
        # [B, N, K*K, C] -> [B, N, C, K*K] -> [B, N, C, K, K]
        mask_feature = self.fc_mask(refined_kernels).permute(0, 1, 3, 2).reshape(B, N, C, 1, 1)

        refined_mask_pred = []
        for i in range(B):
            refined_mask_pred.append(F.conv2d(feature_map[i : i + 1], mask_feature[i]))
        refined_mask_pred = torch.cat(refined_mask_pred, dim=0).reshape(B, N, H, W)

        # Generate class prediction from kernels
        cls_score = self.fc_cls(refined_kernels.sum(-2)).view(B, N, -1)

        # [B, N, K*K, C] -> [B, N, C, K*K] -> [B, N, C, K, K]
        refined_kernels = refined_kernels.permute(0, 1, 3, 2).reshape(B, N, self.in_channels, 1, 1)

        return refined_kernels, refined_mask_pred, cls_score
