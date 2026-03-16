# NormalizationWrapper

Base class for neural network input/output normalization. Registers statistics as PyTorch buffers and handles normalize-model-denormalize in `forward()`.

::: clouds_decoded.normalization.NormalizationWrapper
    options:
      members:
        - forward
        - normalize_input
        - denormalize_output

::: clouds_decoded.normalization.CloudHeightNormalizationWrapper
    options:
      members:
        - forward
