from typing import Callable, Optional

import numpy as np
import torch
from detectron2.projects.point_rend.point_features import point_sample
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast


def linear_sum_assignment_with_nan(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    nan = np.isnan(cost_matrix).any()
    nan_all = np.isnan(cost_matrix).all()
    empty = cost_matrix.size == 0

    if not empty:
        if nan_all:
            print("Matrix contains all NaN values!")
        elif nan:
            print("Matrix contains NaN values!")

        if nan_all:
            cost_matrix = np.empty(shape=(0, 0))
        elif nan:
            cost_matrix[np.isnan(cost_matrix)] = 100

    return linear_sum_assignment(cost_matrix)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the
    best predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        class_cost: Optional[Callable] = None,
        mask_cost: Optional[Callable] = None,
        dice_cost: Optional[Callable] = None,
        class_weight: float = 1.0,
        mask_weight: float = 1.0,
        dice_weight: float = 1.0,
        num_points: int = 0,
    ):
        super().__init__()
        self.class_cost = class_cost
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost

        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight

        assert (
            class_cost is not None or mask_cost is not None or dice_cost is not None
        ), "all cost functions can't be None"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            cost_class, cost_mask, cost_dice = 0, 0, 0
            if self.class_cost is not None and outputs["pred_logits"] is not None:
                cost_class = self.class_weight * self.class_cost(
                    outputs["pred_logits"][b], targets[b]["labels"]
                )

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            if self.num_points > 0:
                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)
            else:
                out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
                tgt_mask = tgt_mask.flatten(1)  # [num_total_targets, H*W]

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()

                if self.mask_cost is not None:
                    cost_mask = self.mask_weight * self.mask_cost(out_mask, tgt_mask)

                if self.dice_cost is not None:
                    cost_dice = self.dice_weight * self.dice_cost(out_mask, tgt_mask)

            # Final cost matrix
            C = cost_class + cost_mask + cost_dice
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment_with_nan(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with pred logits
            "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with pred masks

            targets: List of targets of length batch_size, where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "class_cost: {}".format(self.class_cost),
            "mask_cost: {}".format(self.mask_cost),
            "dice_cost: {}".format(self.dice_cost),
            "class_weight: {}".format(self.class_weight),
            "mask_weight: {}".format(self.mask_weight),
            "dice_weight: {}".format(self.dice_weight),
            "num_points: {}".format(self.num_points),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
