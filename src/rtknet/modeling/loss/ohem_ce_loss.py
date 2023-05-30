import torch
import torch.nn.functional as F

__all__ = ["ohem_ce_loss", "deeplab_ce_loss"]


def ohem_ce_loss(
    inputs, targets, ignore_index=None, ohem_threshold=0.7, n_min=100000, avg_factor=None
):
    loss = (
        F.cross_entropy(inputs, targets, ignore_index=ignore_index, reduction="none")
        .contiguous()
        .view(-1)
    )
    loss, _ = torch.sort(loss, descending=True)
    if loss[n_min] > ohem_threshold:
        loss = loss[loss > ohem_threshold]
    else:
        loss = loss[:n_min]

    if avg_factor is not None:
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)

    return loss


def deeplab_ce_loss(inputs, targets, ignore_index=None, top_k_percent_pixels=0.2, avg_factor=None):
    loss = (
        F.cross_entropy(inputs, targets, ignore_index=ignore_index, reduction="none")
        .contiguous()
        .view(-1)
    )
    top_k_pixels = int(top_k_percent_pixels * loss.numel())
    loss, _ = torch.topk(loss, top_k_pixels)

    if avg_factor is not None:
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)

    return loss
