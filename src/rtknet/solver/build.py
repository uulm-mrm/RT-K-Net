import copy
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from detectron2.solver.build import reduce_param_groups

__all__ = ["get_default_optimizer_params"]


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    weight_decay_embed: Optional[float] = None,
    backbone_lr_factor: Optional[float] = 1.0,
    lr_factor_func: Optional[Callable] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.
    Args:
        model: the model to get the parameters from.
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters.
        weight_decay_embed: override weight decay for torch.nn.Embedding parameters.
        backbone_lr_factor: multiplier of lr for backbone parameters excluding FPN layers.
        lr_factor_func: function to calculate lr decay rate by mapping the parameter names to
            corresponding lr decay rate. Note that setting this option requires
            also setting ``base_lr``.
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.
    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.
    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides
    if lr_factor_func is not None:
        if base_lr is None:
            raise ValueError("lr_factor_func requires base_lr")
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name and "fpn" not in module_name and backbone_lr_factor != 1.0:
                hyperparams["lr"] = hyperparams["lr"] * backbone_lr_factor
            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                print(module_param_name)
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding) and weight_decay_embed is not None:
                hyperparams["weight_decay"] = weight_decay_embed
            if lr_factor_func is not None:
                hyperparams["lr"] *= lr_factor_func(f"{module_name}.{module_param_name}")

            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return reduce_param_groups(params)
