#ifndef ROI_ALIGN_KERNEL_H
#define ROI_ALIGN_KERNEL_H

#include "NvInfer.h"

int roiAlignInference(
    cudaStream_t stream,
    void **outputs,
    const void* const* inputs,
    const int in_width,
    const int in_height,
    const int in_depth,
    const float spatial_scale,
    const int sample_ratio,
    const int pooled_height,
    const int pooled_width,
    const bool position_sensitive,
    const int n_rois);
#endif