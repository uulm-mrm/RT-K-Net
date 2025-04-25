# Adapted from Mask2Former
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/mask_former_panoptic_dataset_mapper.py  # noqa
import copy
import logging

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from panopticapi.utils import rgb2id
from torch.nn import functional as F

from ..transforms import RandomCropWithInstance

__all__ = ["PanopticDatasetMapper"]


class PanopticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        random_flip_id_map,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.random_flip_id_map = random_flip_id_map

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    list(meta.thing_dataset_id_to_contiguous_id.keys()),
                    meta.label_divisor,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "random_flip_id_map": cfg.INPUT.RANDOM_FLIP_ID_MAP,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "PanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if "annotations" in dataset_dict:
            raise ValueError("Panoptic segmentation dataset should not have 'annotations'.")

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Read panoptic segmentation label and segments_info
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Apply augmentations
        aug_input = T.AugInput(image, sem_seg=rgb2id(pan_seg_gt).astype("float32"))
        aug_input, self.transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        pan_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        pan_seg_gt = torch.from_numpy(pan_seg_gt.astype("long"))
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            pan_seg_gt = F.pad(
                pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        # Some classes like arrows are orientation sensitive. If a horizontal flip is performed,
        # class train ids will be swapped accordingly, e.g., left and right arrow labels swap
        flip_train_id = False
        if len(self.random_flip_id_map) > 0:
            for tf in self.transforms:
                if isinstance(tf, T.HFlipTransform):
                    flip_train_id = True

        # Prepare per-category binary masks and semantic segmentation label
        pan_seg_gt = pan_seg_gt.numpy()
        sem_seg_gt = np.full_like(pan_seg_gt, fill_value=self.ignore_label)
        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            mask = pan_seg_gt == segment_info["id"]
            if np.sum(mask) > 1:
                class_id = segment_info["category_id"]
                if flip_train_id:
                    for remap_id_pair in self.random_flip_id_map:
                        if class_id == remap_id_pair[0]:
                            class_id = remap_id_pair[1]
                            break

                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(mask)
                    sem_seg_gt[mask] = class_id

        instances.gt_classes = torch.from_numpy(np.array(classes).astype("long"))
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor

        dataset_dict["sem_seg"] = torch.from_numpy(sem_seg_gt.astype("long"))
        dataset_dict["instances"] = instances

        return dataset_dict
