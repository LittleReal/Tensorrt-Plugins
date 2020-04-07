#include "./roiAlignKernel.h"

#include "NvInfer.h"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)


// The maximum number of blocks to use in the default kernel call. This value is copied from mxnet
constexpr int ROI_MAXIMUM_NUM_BLOCKS = 4096;

// const int kMemUnitBits = 4; The original defination in mxnet is:
//   #if MSHADOW_OLD_CUDA
//   const int kMemUnitBits = 4;
//   const int kMaxThreadsPerBlock = 512;
//   #else
//   const int kMemUnitBits = 5;
//   const int kMaxThreadsPerBlock = 1024;
//   #endif
const int kMaxThreadsPerBlock = 512;
inline int ROI_GET_BLOCKS(const int N) {
    return std::max(
        std::min(
            (N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock,
            ROI_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
}


template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void ROIAlignForwardKernel(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const bool position_sensitive,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int c_unpooled = c;
    int channels_unpooled = channels;
    if (position_sensitive) {
      c_unpooled = c * pooled_height * pooled_width + ph * pooled_width + pw;
      channels_unpooled = channels * pooled_height * pooled_width;
    }
    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels_unpooled + c_unpooled)
        * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;
    top_data[index] = output_val;
  }
}

using namespace std;
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
    const int n_rois){
    const int count=n_rois*in_depth*pooled_height*pooled_width;
    const float* in_data=static_cast<const float*>(inputs[0]);
    const float* roi_data=static_cast<const float*>(inputs[1]);
    float* out_data=static_cast<float*>(outputs[0]);

    ROIAlignForwardKernel<float><<<count, kMaxThreadsPerBlock, 0, stream>>>(
        count,
        in_data,
        spatial_scale,
        position_sensitive,
        in_depth,
        in_height,
        in_width,
        pooled_height,
        pooled_width,
        sample_ratio,
        roi_data,
        out_data);
    return cudaGetLastError() != cudaSuccess;
}

