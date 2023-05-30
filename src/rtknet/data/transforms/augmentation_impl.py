from typing import List

import numpy as np
from detectron2.data.transforms import Augmentation, RandomCrop
from fvcore.transforms import CropTransform

__all__ = ["RandomCropWithInstance"]


class RandomCropWithInstance(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that at least one
    thing category is present in panoptic segmentation ground truth.
    The function attempts to find such a valid cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        thing_ids: List[int] = None,
        label_divisor: int = None,
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
            thing_ids: List of thing category ids
            label_divisor: label_divisor used to encode panoptic instances
        """
        self.crop_aug = RandomCrop(crop_type, crop_size)
        self._init(locals())

    def get_transform(self, image, sem_seg):
        """
        Args:
            image: image
            sem_seg: panoptic ground truth in uint16 format.
        """
        h, w = sem_seg.shape
        for _ in range(10):
            crop_size = self.crop_aug.get_crop_size((h, w))
            y0 = np.random.randint(h - crop_size[0] + 1)
            x0 = np.random.randint(w - crop_size[1] + 1)
            sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
            labels = np.unique(sem_seg_temp.astype(np.long))
            if not (set(labels // self.label_divisor) & set(self.thing_ids)):
                continue
            found_instance = False
            for id in labels:
                if id // self.label_divisor not in self.thing_ids:
                    continue
                mask_index = np.where(sem_seg == id)
                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                y, x = int(np.round(center_y)), int(np.round(center_x))
                if y0 <= y < y0 + crop_size[0] and x0 <= x < x0 + crop_size[1]:
                    found_instance = True
                    break
            if found_instance:
                break

        crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
        return crop_tfm
