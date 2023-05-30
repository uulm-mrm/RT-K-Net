import torch
import torch.nn.functional as F

__all__ = ["sigmoid_focal_loss"]


def sigmoid_focal_loss(
    inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2, avg_factor=None
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification
            label for each element in inputs (0 for the negative class, 1 for the positive class).
        weight: A weight tensor to weight each element of the loss
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        avg_factor: Average factor that is used to average the loss. Defaults to None.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    num_classes = inputs.size(1)
    targets = F.one_hot(targets, num_classes=num_classes + 1)
    targets = targets[:, :num_classes].float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight

    if avg_factor is not None:
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)

    return loss
