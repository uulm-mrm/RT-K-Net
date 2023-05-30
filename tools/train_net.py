#!/usr/bin/env python3
import itertools
import os
import warnings
from datetime import datetime

import torch
import torch.backends.cudnn
import torch.cuda.amp
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.engine.defaults import _try_get_key
from detectron2.evaluation import COCOPanopticEvaluator, DatasetEvaluators
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.utils import comm
from detectron2.utils.events import EventStorage, JSONWriter
from detectron2.utils.logger import setup_logger
from detectron2.utils.registry import locate
from rtknet import add_rtknet_config
from rtknet.data import register_mapillary_vistas_panoptic  # noqa
from rtknet.evaluation import TensorboardImageWriter
from rtknet.solver import get_default_optimizer_params


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = [
            TensorboardImageWriter(dataset_name, cfg.OUTPUT_DIR, cfg.TEST.EVAL_PERIOD)
        ]
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if (
            evaluator_type == "cityscapes_panoptic_seg"
            or evaluator_type == "mapillary_vistas_panoptic_seg"
        ):
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if len(evaluator_list) == 1:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = locate(cfg.INPUT.TRAIN_DATASET_MAPPER)(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = locate(cfg.INPUT.TEST_DATASET_MAPPER)(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Overwrite build_lr_scheduler to call deeplab build_lr_scheduler function.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
            backbone_lr_factor=cfg.SOLVER.BACKBONE_MULTIPLIER,
        )

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping for now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_full_model_gradient_clipping(torch.optim.Adam)(
                params, cfg.SOLVER.BASE_LR
            )
        elif optimizer_type == "ADAMW":
            return maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Overwrite default test function with disabled cudnn benchmark and added amp autocast
        """
        torch.backends.cudnn.benchmark = False
        with torch.cuda.amp.autocast(enabled=cfg.TEST.AMP.ENABLED):
            results = super().test(cfg, model, evaluators)

        # Restore original setting
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )

        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_rtknet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.WRITE_OUTPUT_TO_SUBDIR:
        config_file_name = args.config_file.split("/")[-1].replace(".yaml", "")
        cfg.OUTPUT_DIR = os.path.join(
            cfg.OUTPUT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + config_file_name
        )
        if args.eval_only:
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "-EvalOnly"
    # Download google drive weights
    if "drive.google.com" in cfg.MODEL.WEIGHTS:
        import gdown

        gdown.download(cfg.MODEL.WEIGHTS, "/tmp/init.pth", quiet=False, fuzzy=True)
        cfg.MODEL.WEIGHTS = "/tmp/init.pth"
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "rtknet" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="rtknet")

    # cudnn deterministic setting should be enabled when training with manual seed
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.deterministic = _try_get_key(cfg, "CUDNN_DETERMINISTIC", default=False)

    return cfg


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = setup(args)

    if args.eval_only:
        with EventStorage(cfg.SOLVER.MAX_ITER) as storage:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
            if res:
                flattened_results = flatten_results_dict(res)
                storage.put_scalars(**flattened_results, smoothing_hint=False)
            writer = JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))
            writer.write()
            return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
