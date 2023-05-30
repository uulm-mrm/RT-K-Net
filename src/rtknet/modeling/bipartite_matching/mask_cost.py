import torch
import torch.nn.functional as F

__all__ = ["sigmoid_ce_cost", "sigmoid_ce_cost_jit"]


def sigmoid_ce_cost(
    inputs: torch.Tensor, targets: torch.Tensor, omit_log: bool = False, clamp_sigmoid: bool = False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification
            label for each element in inputs (0 for the negative class, 1 for the positive class).
        omit_log: A bool. Whether to omit the log in loss calculation. Log can be omitted for
            matching cost calculation since it does not change the relation of the costs.
        clamp_sigmoid: If True, inputs are clamped after sigmoid to [0.01, 1.0]
    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    hw = inputs.shape[1]

    if omit_log:
        inputs = inputs.sigmoid()
        if clamp_sigmoid:
            inputs = inputs.clamp(min=0.01, max=1.0)
        pos = torch.einsum("nc,mc->nm", inputs, targets)
        neg = torch.einsum("nc,mc->nm", 1 - inputs, 1 - targets)
        loss = -(pos + neg)
    else:
        pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
        neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )

    return loss / hw


sigmoid_ce_cost_jit = torch.jit.script(sigmoid_ce_cost)  # type: torch.jit.ScriptModule
