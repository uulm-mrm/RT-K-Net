import torch

__all__ = ["focal_loss_class_cost"]


def focal_loss_class_cost(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: int = 2,
    eps: float = 1e-12,
):
    inputs = inputs.sigmoid()
    pos = -alpha * torch.log(inputs + eps) * torch.pow(1 - inputs, gamma)
    neg = -(1 - alpha) * torch.log(1 - inputs + eps) * torch.pow(inputs, gamma)

    cls_cost = pos[:, targets] - neg[:, targets]
    return cls_cost
