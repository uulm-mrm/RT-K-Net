from functools import partial

import torch
import torch.nn.functional as F
from rtknet.modeling.bipartite_matching import (
    HungarianMatcher,
    dice_cost_jit,
    focal_loss_class_cost,
    sigmoid_ce_cost_jit,
)
from rtknet.modeling.loss import deeplab_ce_loss, dice_loss_jit, sigmoid_focal_loss
from rtknet.utils import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from torch import nn
from torch.cuda.amp import custom_fwd
from torch.distributed import all_reduce, get_world_size

__all__ = ["RTKNetCriterion"]


class RTKNetCriterion(nn.Module):
    def __init__(
        self,
        num_proposals,
        num_thing_classes,
        num_stuff_classes,
        num_classes,
        ignore_label,
        rank_weight,
        seg_weight,
        mask_weight,
        dice_weight,
        cls_weight,
        inst_disc_weight,
    ):
        super().__init__()

        self.num_proposals = num_proposals
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        self.rank_weight = rank_weight
        self.seg_weight = seg_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.cls_weight = cls_weight
        self.inst_disc_weight = inst_disc_weight

        self.loss_rank = partial(F.cross_entropy, reduction="none")
        self.loss_seg = deeplab_ce_loss
        self.loss_mask = F.binary_cross_entropy_with_logits
        self.loss_dice = dice_loss_jit
        self.loss_cls = sigmoid_focal_loss
        self.loss_inst_disc = nn.CrossEntropyLoss()

        self.matcher = HungarianMatcher(
            mask_cost=partial(sigmoid_ce_cost_jit, omit_log=True, clamp_sigmoid=True),
            dice_cost=partial(dice_cost_jit, omit_constant=True, clamp_sigmoid=True),
            class_cost=focal_loss_class_cost,
            mask_weight=mask_weight,
            dice_weight=dice_weight,
            class_weight=cls_weight,
        )

    def forward(self, outputs, targets):
        instance_targets = targets["instances"]

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = self._get_avg_num_masks(instance_targets)

        # Auxiliary seg loss
        losses = self.seg_loss(outputs["seg_mask_pred"], targets["sem_seg"])
        if self.inst_disc_weight > 0:
            losses.update(
                self.instance_discrimination_loss(
                    outputs["feature_map"], [t["masks"] for t in instance_targets]
                )
            )

        # Calculate losses for iter heads
        for stage in range(len(outputs["outputs"])):
            mask_pred = outputs["outputs"][stage]["pred_masks"]
            cls_score = outputs["outputs"][stage]["pred_logits"]
            prediction = {"pred_masks": mask_pred, "pred_logits": cls_score}
            indices = self.matcher(prediction, instance_targets)

            single_stage_loss = {}
            if cls_score is not None:
                single_stage_loss.update(
                    self.cls_loss(cls_score, instance_targets, indices, num_masks)
                )
            single_stage_loss.update(
                self.mask_loss(
                    mask_pred,
                    [t["masks"] for t in instance_targets],
                    indices,
                )
            )
            for key, value in single_stage_loss.items():
                losses[f"s{stage + 1}_{key}"] = value

        return losses

    def cls_loss(self, cls_pred, targets, indices, num_masks):
        N, C, _ = cls_pred.shape

        idx = self._get_src_permutation_idx(indices)
        labels = torch.full(
            cls_pred.shape[:2], self.num_classes, dtype=torch.int64, device=cls_pred.device
        )
        labels[idx] = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        return {
            "loss_cls": self.cls_weight
            * self.loss_cls(
                cls_pred.view(N * C, -1), labels.view(-1), weight=None, avg_factor=num_masks
            )
        }

    def mask_loss(self, mask_pred, mask_targets, indices):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = mask_pred[src_idx]
        target_masks, _ = nested_tensor_from_tensor_list(mask_targets).decompose()
        target_masks = target_masks.to(src_masks)
        pos_target_masks = target_masks[tgt_idx]

        mask_losses = {
            "loss_mask": self.mask_weight * self.loss_mask(src_masks, pos_target_masks),
            "loss_dice": self.dice_weight * self.loss_dice(src_masks, pos_target_masks).mean(),
        }

        if self.loss_rank is not None:
            N, C, H, W = mask_pred.shape
            mask_targets = target_masks.new_zeros(N, C, H, W)
            mask_targets[src_idx] = pos_target_masks
            batch_mask_targets = mask_targets.view(N, -1, H, W)
            # We use argmax to calculate the rank target. Append a mask with all 0.5 values to get
            # ignored pixels which are not present in any target mask.
            rank_target = (
                torch.cat(
                    [batch_mask_targets, batch_mask_targets.new_full((N, 1, H, W), 0.5)], dim=1
                )
                .argmax(1)
                .long()
            )

            mask_losses["loss_rank"] = (
                self.rank_weight
                * self.loss_rank(mask_pred, rank_target, ignore_index=rank_target.max()).mean()
            )

        return mask_losses

    def seg_loss(self, seg_pred, seg_target):
        num_dense_pos = (seg_target >= 0) & (seg_target < self.num_classes)
        num_dense_pos = num_dense_pos.sum().float().clamp(min=1.0)
        return {
            "loss_seg": self.seg_weight
            * self.loss_seg(
                seg_pred, seg_target, avg_factor=num_dense_pos, ignore_index=self.ignore_label
            )
        }

    @custom_fwd(cast_inputs=torch.float32)
    def instance_discrimination_loss(self, feature_map, mask_targets):
        feature_map = F.normalize(feature_map)

        target_masks, _ = nested_tensor_from_tensor_list(mask_targets).decompose()
        N, C, H, W = target_masks.shape
        target_masks = torch.round(target_masks.view(N, C, H * W))
        target_masks_area = target_masks.sum(-1)

        target_masks_area = torch.einsum("bn,bnp->bp", target_masks_area, target_masks)
        target_masks_area[target_masks_area == 0] = 1
        inverse_target_masks_area = H * W / target_masks_area

        pixel_gt_non_void_mask = inverse_target_masks_area.new_ones(N, H * W)
        pixel_gt_non_void_mask[target_masks.sum(1) == 0] = 0

        # Compute pixel space sampling indices according to mask area.
        instance_disc_sample_temperature = 0.3
        sample_logits = torch.log(inverse_target_masks_area) * instance_disc_sample_temperature

        # A large negative constant applied before softmax. This will make the softmax
        # ignore the masked logits.
        _SOFTMAX_MASKING_CONSTANT = -99999.0
        sample_logits += (1 - pixel_gt_non_void_mask) * _SOFTMAX_MASKING_CONSTANT

        instance_discrimination_sample_k = 4096
        gumbel_noise = -torch.log(
            -torch.log(torch.rand(sample_logits.shape, device=sample_logits.device))
        )
        _, sample_indices = torch.topk(
            sample_logits + gumbel_noise, instance_discrimination_sample_k
        )

        pixel_gt_sampled_feature = torch.gather(
            target_masks, index=sample_indices.unsqueeze(1).repeat(1, C, 1), dim=-1
        )

        sampled_gt_similarity = torch.einsum(
            "bik,bij->bkj", pixel_gt_sampled_feature, pixel_gt_sampled_feature
        )

        C = feature_map.shape[1]
        feature_map = feature_map.view(N, C, H * W)
        pixel_pred_sampled_feature = torch.gather(
            feature_map, index=sample_indices.unsqueeze(1).repeat(1, C, 1), dim=-1
        )

        sampled_pred_similarity = torch.einsum(
            "bik,bij->bkj", pixel_pred_sampled_feature, pixel_pred_sampled_feature
        )

        pixel_normalizing_constant = torch.sum(sampled_gt_similarity, dim=-1, keepdim=True)
        pixel_normalizing_constant[pixel_normalizing_constant == 0] = 1
        sampled_gt_similarity /= pixel_normalizing_constant
        sampled_pred_similarity /= instance_disc_sample_temperature

        return {
            "loss_inst_disc": self.inst_disc_weight
            * self.loss_inst_disc(sampled_pred_similarity, sampled_gt_similarity)
        }

    @staticmethod
    def _get_avg_num_masks(targets):
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            num_masks, dtype=torch.float, device=targets[0]["labels"].device
        )
        if is_dist_avail_and_initialized():
            all_reduce(num_masks)
            num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        return num_masks

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
