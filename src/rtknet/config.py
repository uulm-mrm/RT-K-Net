from detectron2.config import CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config

__all__ = ["add_rtknet_config"]


def add_rtknet_config(cfg):
    add_deeplab_config(cfg)

    # General parameters.
    # Sets torch.backends.cudnn.deterministic variable to increase determinism.
    cfg.CUDNN_DETERMINISTIC = False
    # If true, a subdir in cfg.OUTPUT_DIR is created based on current time and config.
    # All output is redirected to the newly created OUTPUT_DIR
    cfg.WRITE_OUTPUT_TO_SUBDIR = True

    # input config
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = True
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    # Specify dataset mappers. String needs to contain the package path as well as the class name.
    cfg.INPUT.TRAIN_DATASET_MAPPER = "detectron2.data.DatasetMapper"
    cfg.INPUT.TEST_DATASET_MAPPER = "detectron2.data.DatasetMapper"
    # List of train id tuples which will be switched in case a lr-flip augmentation is performed,
    # e.g., left and right arrow class ids will be switched with each other
    cfg.INPUT.RANDOM_FLIP_ID_MAP = []

    # solver config
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = None
    # lr multiplier for backbone layers
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # model config
    # Pad images and GT inside batch to ensure that minimum divisibility of model is met
    cfg.MODEL.SIZE_DIVISIBILITY = 32

    # RTFormer backbone config
    cfg.MODEL.RT_FORMER_BACKBONE = CN()
    cfg.MODEL.RT_FORMER_BACKBONE.VARIANT = ""
    cfg.MODEL.RT_FORMER_BACKBONE.NORM = "BN"

    cfg.MODEL.FEATURE_MAP_GENERATOR = CN()
    cfg.MODEL.FEATURE_MAP_GENERATOR.NAME = "RTFormerHead"
    cfg.MODEL.FEATURE_MAP_GENERATOR.INIT_FUNC = "rtknet.layers.weight_init.kaiming_init"
    cfg.MODEL.FEATURE_MAP_GENERATOR.OUT_CHANNELS = 256

    # RTFormer head config
    cfg.MODEL.RT_FORMER_HEAD = CN()
    cfg.MODEL.RT_FORMER_HEAD.BASE_CHANNELS = 64
    cfg.MODEL.RT_FORMER_HEAD.HEAD_CHANNELS = 128
    cfg.MODEL.RT_FORMER_HEAD.NORM = "BN"

    cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.SEM_SEG_HEAD.NORMALIZE_SIGMOID_MASKS = True
    cfg.MODEL.SEM_SEG_HEAD.NUM_KERNEL_UPDATE_HEADS = 3
    cfg.MODEL.SEM_SEG_HEAD.NORM = ""
    cfg.MODEL.SEM_SEG_HEAD.INIT_FUNC = "rtknet.layers.weight_init.kaiming_init"

    # kernel update heads config
    cfg.MODEL.KERNEL_UPDATE_HEADS = CN()
    cfg.MODEL.KERNEL_UPDATE_HEADS.KERNEL_FEATURE_CHANNELS = 256
    cfg.MODEL.KERNEL_UPDATE_HEADS.OUT_CHANNELS = 256

    # set training config
    cfg.MODEL.SET_CRITERION = CN()
    # loss weights
    cfg.MODEL.SET_CRITERION.RANK_WEIGHT = 0.1
    cfg.MODEL.SET_CRITERION.SEG_WEIGHT = 1.0
    cfg.MODEL.SET_CRITERION.MASK_WEIGHT = 1.0
    cfg.MODEL.SET_CRITERION.DICE_WEIGHT = 4.0
    cfg.MODEL.SET_CRITERION.CLASS_WEIGHT = 2.0
    cfg.MODEL.SET_CRITERION.INST_DISC_WEIGHT = 1.0

    # model inference config
    cfg.MODEL.TEST = CN()
    cfg.MODEL.TEST.INSTANCE_SCORE_THRESHOLD = 0.3
    cfg.MODEL.TEST.OVERLAP_THRESHOLD = 0.6

    # Test parameters
    cfg.TEST.AMP = CN()
    cfg.TEST.AMP.ENABLED = True
