# Cloud Mask

Binary cloud mask using SegFormer-B2 deep learning (4-class inference, then binarized) or simple reflectance thresholding.

## Processors

::: clouds_decoded.modules.cloud_mask.processor.CloudMaskProcessor
    options:
      members:
        - process

::: clouds_decoded.modules.cloud_mask.processor.ThresholdCloudMaskProcessor
    options:
      members:
        - process
