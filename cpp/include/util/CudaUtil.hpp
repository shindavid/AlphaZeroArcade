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
int cuda_device_to_ordinal(const std::string& cuda_device);

cudaStream_t create_stream();
void destroy_stream(cudaStream_t stream);
void synchronize_stream(cudaStream_t stream);
void set_device(int device_id);

void cpu2gpu_memcpy(void* dst, const void* src, size_t n_bytes);
void gpu2cpu_memcpy(void* dst, const void* src, size_t n_bytes);
void cpu2gpu_memcpy_async(cudaStream_t stream, void* dst, const void* src, size_t n_bytes);
void gpu2cpu_memcpy_async(cudaStream_t stream, void* dst, const void* src, size_t n_bytes);

void* gpu_malloc(size_t n_bytes);
void* cpu_malloc(size_t n_bytes);
void gpu_free(void* ptr);
void cpu_free(void* ptr);
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
