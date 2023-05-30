from detectron2.utils.registry import Registry

__all__ = ["FEATURE_MAP_GENERATOR_REGISTRY", "build_feature_map_generator"]


FEATURE_MAP_GENERATOR_REGISTRY = Registry("FEATURE_MAP_GENERATOR_REGISTRY")
FEATURE_MAP_GENERATOR_REGISTRY.__doc__ = """
Registry for feature map generators, which extract feature maps from backbone features
"""


def build_feature_map_generator(cfg, input_shape=None):
    """
    Build a feature map generator from `cfg.MODEL.FEATURE_MAP_GENERATOR.NAME`.
    """
    feature_map_generator_name = cfg.MODEL.FEATURE_MAP_GENERATOR.NAME
    feature_map = FEATURE_MAP_GENERATOR_REGISTRY.get(feature_map_generator_name)(cfg, input_shape)
    return feature_map
