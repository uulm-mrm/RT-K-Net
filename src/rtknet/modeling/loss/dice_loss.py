import torch

__all__ = ["dice_loss", "dice_loss_jit"]


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    naive_dice: bool = False,
    smooth: float = 0.0,
    eps: float = 2e-3,
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
        smooth: A float value, which is added to both nominator and denominator. Can be used to
            handle cases, where the ground truth has very few white (or no) positive class labels.
            Defaults to 0.
        eps: A float. Avoid dividing by zero. Defaults to 2e-3.
    Returns:
        Cost tensor
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * torch.sum(inputs * targets, 1)
    if naive_dice:
        denominator = torch.sum(inputs, 1) + torch.sum(targets, 1)
    else:
        denominator = torch.sum(inputs * inputs, 1) + torch.sum(targets * targets, 1)

    loss = 1 - (numerator + smooth) / (denominator + smooth + eps)
    return loss


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule
