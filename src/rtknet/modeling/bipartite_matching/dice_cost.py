import torch

__all__ = ["dice_cost", "dice_cost_jit"]


def dice_cost(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    naive_dice: bool = False,
    omit_constant: bool = False,
    smooth: float = 0.0,
    eps: float = 2e-3,
    clamp_sigmoid: bool = False,
):
    """
    Compute the DICE matching cost, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification
            label for each element in inputs (0 for the negative class, 1 for the positive class).
        naive_dice: A bool. If false, use the dice loss defined in the V-Net paper, otherwise, use
            the naive dice loss in which the power of the number in the denominator is the first
            power instead of the second power. Defaults to False.
        omit_constant: A bool to specify whether the constant of 1 will be omitted or not. Can be
            done for matching cost calculation since it does not change the relation of the costs.
        smooth: A float value, which is added to both nominator and denominator. Can be used to
            handle cases, where the ground truth has very few white (or no) positive class labels.
            Defaults to 0.
        eps: A float. Avoid dividing by zero. Defaults to 2e-3.
        clamp_sigmoid: If True, inputs are clamped after sigmoid to [0.001, 1.0]
    Returns:
        Cost tensor
    """
    inputs = inputs.sigmoid()
    if clamp_sigmoid:
        inputs = inputs.clamp(min=0.001, max=1.0)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    if naive_dice:
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    else:
        denominator = torch.pow(inputs, 2).sum(-1)[:, None] + torch.pow(targets, 2).sum(-1)[None, :]

    if omit_constant:
        loss = -(numerator + smooth) / (denominator + smooth + eps)
    else:
        loss = 1 - (numerator + smooth) / (denominator + smooth + eps)

    return loss


dice_cost_jit = torch.jit.script(dice_cost)  # type: torch.jit.ScriptModule
