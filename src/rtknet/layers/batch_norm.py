# Adapted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py  # noqa
# but using nn.LayerNorm as LN variant
from detectron2.layers import BatchNorm2d, FrozenBatchNorm2d, NaiveSyncBatchNorm
from detectron2.utils import env
from torch import nn

__all__ = ["get_norm"]


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": nn.LayerNorm,
        }[norm]
    return norm(out_channels, **kwargs)
