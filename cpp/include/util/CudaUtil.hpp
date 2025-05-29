#pragma once

#include <util/Exception.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <format>
#include <string>

namespace cuda_util {

void dump_memory_info();

// "cuda:1" -> 1
// "1" -> 1
int get_device_id(const std::string& cuda_device);

void gpu2cpu_memcpy(void* dst, const void* src, size_t n_bytes);
void* gpu_malloc(size_t n_bytes);
void assert_on_device(int device_id);

// Validate that the batch size is within the dynamic range of the given tensor in the engine.
void validate_batch_size(nvinfer1::ICudaEngine* engine, const char* tensor_name, int profile_index,
                         int batch_size);

// Calculate the total number of elements in a Dims object, i.e., the product of all dimensions.
size_t dim_size(const nvinfer1::Dims& dims);

}  // namespace cuda_util

// This allows us to pass an object of type nvinfer1::Dims to std::format
template <>
struct std::formatter<nvinfer1::Dims> : std::formatter<std::string> {
  auto format(const nvinfer1::Dims& dims, format_context& ctx) const {
    auto out = ctx.out();
    *out++ = '[';
    for (int i = 0; i < dims.nbDims; ++i) {
      if (i) *out++ = ',';
      // format each dimension with the default integer formatter
      out = std::format_to(out, "{}", dims.d[i]);
    }
    *out++ = ']';
    return out;
  }
};

#include <inline/util/CudaUtil.inl>
