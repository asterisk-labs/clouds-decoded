"""
Unified access point for Clouds Decoded processors.
This module re-exports processors and configurations from the installed sub-modules,
allowing them to be accessed from a central `clouds_decoded.processors` namespace.
"""

import logging

logger = logging.getLogger(__name__)

# Cloud Height Module
from clouds_decoded.modules.cloud_height import CloudHeightProcessor, CloudHeightConfig

# Cloud Mask Module
from clouds_decoded.modules.cloud_mask import (
    CloudMaskProcessor,
    ThresholdCloudMaskProcessor,
    CloudMaskConfig,
    PostProcessParams
)

# Albedo Estimator Module
from clouds_decoded.modules.albedo_estimator import AlbedoEstimator, AlbedoEstimatorConfig

# Refl2Prop (Cloud Properties Inversion) Module
try:
    from clouds_decoded.modules.refl2prop.processor import CloudPropertyInverter
except ImportError:
    logger.debug("Refl2Prop module not found. CloudPropertyInverter not available (requires PyTorch).")
    CloudPropertyInverter = None

__all__ = [
    'CloudHeightProcessor',
    'CloudHeightConfig',
    'CloudMaskProcessor',
    'ThresholdCloudMaskProcessor',
    'CloudMaskConfig',
    'PostProcessParams',
    'AlbedoEstimator',
    'AlbedoEstimatorConfig',
    'CloudPropertyInverter',
]
