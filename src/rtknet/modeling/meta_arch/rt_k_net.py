from functools import partial
from typing import Callable, Tuple

import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList
from rtknet.modeling.criterion import RTKNetCriterion
from rtknet.modeling.post_processing import get_prediction
from torch import nn
from torch.nn import functional as F


@META_ARCH_REGISTRY.register()
class RTKNet(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        inference_func: Callable,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer(
            "pixel_mean", torch.tensor([x for x in pixel_mean]).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor([x for x in pixel_std]).view(-1, 1, 1), False
        )
        self.inference = inference_func

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        criterion = RTKNetCriterion(
            num_proposals=cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES,
            num_thing_classes=len(meta.thing_dataset_id_to_contiguous_id.values()),
            num_stuff_classes=len(meta.stuff_dataset_id_to_contiguous_id.values()),
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            ignore_label=meta.ignore_label,
            rank_weight=cfg.MODEL.SET_CRITERION.RANK_WEIGHT,
            seg_weight=cfg.MODEL.SET_CRITERION.SEG_WEIGHT,
            mask_weight=cfg.MODEL.SET_CRITERION.MASK_WEIGHT,
            dice_weight=cfg.MODEL.SET_CRITERION.DICE_WEIGHT,
            cls_weight=cfg.MODEL.SET_CRITERION.CLASS_WEIGHT,
            inst_disc_weight=cfg.MODEL.SET_CRITERION.INST_DISC_WEIGHT,
        )

        inference_func = partial(
            get_prediction,
            num_classes=sem_seg_head.num_classes,
            last_stuff_id=max(meta.stuff_dataset_id_to_contiguous_id.values()),
            label_divisor=meta.label_divisor,
            instance_score_threshold=cfg.MODEL.TEST.INSTANCE_SCORE_THRESHOLD,
            overlap_threshold=cfg.MODEL.TEST.OVERLAP_THRESHOLD,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "inference_func": inference_func,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        inputs, outputs = {}, {}
        images = [
            (x["image"].to(self.device).float() - self.pixel_mean) / self.pixel_std
            for x in batched_inputs
        ]
        images = ImageList.from_tensors(images, self.size_divisibility)
        inputs["image"] = images.tensor

        features = self.backbone(inputs["image"])
        outputs.update(self.sem_seg_head(features))

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            return losses

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        del outputs

        processed_results = []
        for (mask_cls_result, mask_pred_result, input_per_image, image_size,) in zip(
            mask_cls_results,
            mask_pred_results,
            batched_inputs,
            images.image_sizes,
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = F.interpolate(
                mask_pred_result.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False
            )[0]
            mask_cls_result = mask_cls_result.to(mask_pred_result)

            # panoptic segmentation inference
            panoptic_r = self.inference(mask_cls_result, mask_pred_result)
            panoptic_r = (
                F.interpolate(
                    panoptic_r.expand(1, 1, -1, -1).float(), size=(height, width), mode="nearest"
                )
                .squeeze()
                .long()
            )
            processed_results[-1]["panoptic_seg"] = (panoptic_r, None)

        return processed_results

    def prepare_targets(self, batched_inputs, images):
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        h_pad, w_pad = images.tensor.shape[-2:]
        instances = []
        for targets_per_image in gt_instances:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            assign_H = padded_masks.shape[1] // 4
            assign_W = padded_masks.shape[2] // 4
            if padded_masks.shape[0] != 0:
                padded_masks = F.interpolate(
                    padded_masks[None].float(),
                    (assign_H, assign_W),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            else:
                padded_masks = torch.zeros(
                    (gt_masks.shape[0], assign_H, assign_W),
                    dtype=gt_masks.dtype,
                    device=gt_masks.device,
                )

            instances.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        targets = {"instances": instances}

        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(
            gt_sem_seg, self.size_divisibility, pad_value=self.metadata.ignore_label
        ).tensor
        gt_sem_seg = F.interpolate(
            gt_sem_seg[None].float(),
            scale_factor=0.25,
            mode="nearest",
        )[0].long()
        targets["sem_seg"] = gt_sem_seg

        return targets
