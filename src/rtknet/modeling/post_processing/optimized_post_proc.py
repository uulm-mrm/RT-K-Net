import torch
import torch.nn.functional as F

__all__ = ["get_prediction", "merge_cls_scores"]


def get_prediction(
    mask_cls,
    mask_pred,
    num_classes,
    last_stuff_id,
    label_divisor,
    instance_score_threshold,
    overlap_threshold,
    use_sigmoid=True,
):
    score_pred, label_pred = merge_cls_scores(mask_cls, use_sigmoid=use_sigmoid)

    keep = label_pred.ne(num_classes) & (
        score_pred >= min(instance_score_threshold, score_pred.max().item())
    )

    score_pred = score_pred[keep]
    label_pred = label_pred[keep]
    mask_pred = mask_pred[keep].sigmoid()

    instance_ids = torch.arange(mask_pred.shape[0], dtype=torch.int32, device=mask_pred.device)
    instance_ids[label_pred <= last_stuff_id] = 0  # set stuff instance ids to 0
    # encode instance ids. Add one for einsum used later
    label_pred = label_pred * label_divisor + instance_ids + 1

    cur_prob_masks = score_pred.view(-1, 1, 1) * mask_pred

    cur_mask_ids = cur_prob_masks.argmax(0, keepdim=True)
    one_hot = torch.zeros_like(mask_pred, dtype=torch.bool)
    one_hot.scatter_(0, cur_mask_ids, 1)
    original_mask = mask_pred >= 0.5
    mask = one_hot  # & original_mask

    original_area = torch.sum(original_mask, dim=(1, 2)) * overlap_threshold
    mask_area = torch.sum(one_hot, dim=(1, 2))

    keep2 = (mask_area >= original_area) & (mask_area > 0) & (original_area > 0)
    mask = mask[keep2].float()
    label_pred = label_pred[keep2].float()

    with torch.cuda.amp.autocast(False):
        panoptic_seg = torch.einsum("q,qhw->hw", label_pred, mask).int()
    panoptic_seg -= 1  # subtract one to get original ids

    return panoptic_seg


def merge_cls_scores(mask_cls, use_sigmoid=True):
    if use_sigmoid:
        mask_cls = mask_cls.sigmoid()
    else:
        mask_cls = F.softmax(mask_cls, dim=-1)

    scores, labels = mask_cls.max(dim=1)
    return scores, labels
