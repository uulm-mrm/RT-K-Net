# Adapted from Mask2Former
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/utils/misc.py
import torch.distributed as dist

__all__ = ["is_dist_avail_and_initialized"]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
