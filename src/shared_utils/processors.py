"""
Unified access point for Clouds Decoded processors.
This module re-exports processors and configurations from the installed sub-modules,
allowing them to be accessed from a central `clouds_decoded.processors` namespace.
"""

import logging

logger = logging.getLogger(__name__)

# Cloud Height Module
try:
    from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
except ImportError:
    logger.debug("Cloud Height module not found. CloudHeightProcessor not available.")
    CloudHeightProcessor = None
    CloudHeightConfig = None

# Refl2Prop (Cloud Properties Inversion) Module
try:
    from clouds_decoded.modules.refl2prop.processor import CloudPropertyInverter
except ImportError:
    logger.debug("Refl2Prop module not found. CloudPropertyInverter not available.")
    CloudPropertyInverter = None

# Albedo Estimator Module
try:
    from clouds_decoded.modules.albedo_estimator.processor import AlbedoEstimator
except ImportError:
    logger.debug("Albedo Estimator module not found. AlbedoEstimator not available.")
    AlbedoEstimator = None

# Cloud Mask Module
try:
    from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor, ThresholdCloudMaskProcessor
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig, PostProcessParams
except ImportError:
    logger.debug("Cloud Mask module not found. CloudMaskProcessor not available.")
    CloudMaskProcessor = None
    ThresholdCloudMaskProcessor = None
    CloudMaskConfig = None
    PostProcessParams = None
