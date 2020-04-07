## Tensorrt Common plugins
Tensorrt support adding common plugins that not contained by itself. The rule is that inherint IPluginV2 and IPluginCreator refer to NvInferRuntimecommon.h[https://github.com/NVIDIA/TensorRT/blob/master/include/NvInferRuntimeCommon.h].
