#!/usr/bin/env python3
import argparse
import os

import cv2
import matplotlib.colors as mplc
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.utils.registry import locate
from detectron2.utils.visualizer import Visualizer
from rtknet.config import add_rtknet_config


def setup(args):
    cfg = get_cfg()
    add_rtknet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", required=True, help="path to config file")
    parser.add_argument("--scale", type=float, default=1.0, help="image scale for visualizations")
    parser.add_argument(
        "--write-images", action="store_true", help="whether to write images to files"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    mapper = locate(cfg.INPUT.TRAIN_DATASET_MAPPER)(cfg, is_train=True)
    train_data_loader = build_detection_train_loader(cfg, mapper=mapper)
    for batch in train_data_loader:
        for per_image in batch:
            print(f"Visualize sample {per_image['file_name']}")
            out_file = os.path.splitext(os.path.basename(per_image["file_name"]))[0]

            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

            visualizer = Visualizer(img, metadata=metadata, scale=args.scale)
            if args.write_images:
                cv2.imwrite(f"{out_file}.jpg", visualizer.get_output().get_image()[:, :, ::-1])
            else:
                cv2.imshow(f"image", visualizer.get_output().get_image()[:, :, ::-1])
            if "sem_seg" in per_image:
                vis = visualizer.draw_sem_seg(per_image["sem_seg"], alpha=0.5)
                if args.write_images:
                    cv2.imwrite(f"{out_file}_sem.jpg", vis.get_image()[:, :, ::-1])
                else:
                    cv2.imshow(f"semantic seg", vis.get_image()[:, :, ::-1])

            if "instances" in per_image:
                colors = []
                for cat in per_image["instances"].gt_classes:
                    color = mplc.to_rgb(tuple(x / 255 for x in metadata.stuff_colors[cat]))
                    # Jitter thing cat colors
                    is_thing = False
                    if cat in list(metadata.thing_dataset_id_to_contiguous_id.values()):
                        is_thing = True
                    # Fix for cityscapes_panoptic_separated
                    try:
                        if cat in list(metadata.thing_train_id2contiguous_id.keys()):
                            is_thing = True
                            color = mplc.to_rgb(
                                tuple(
                                    x / 255
                                    for x in metadata.stuff_colors[
                                        metadata.thing_train_id2contiguous_id[cat.item()]
                                    ]
                                )
                            )
                    except AttributeError:
                        pass

                    if is_thing:
                        vec = np.random.rand(3)
                        # better to do it in another color space
                        vec = vec / np.linalg.norm(vec) * 0.5
                        color = np.clip(vec + color, 0, 1)
                    colors.append(color)

                vis = visualizer.overlay_instances(
                    masks=per_image["instances"].gt_masks, assigned_colors=colors
                )
                if args.write_images:
                    cv2.imwrite(f"{out_file}_inst.jpg", vis.get_image()[:, :, ::-1])
                else:
                    cv2.imshow(f"instances", vis.get_image()[:, :, ::-1])

            k = cv2.waitKey()
            if k == 27 or k == 113:  # Esc or q key to stop
                exit(0)
            if args.write_images:
                input("Press Enter to continue...")
