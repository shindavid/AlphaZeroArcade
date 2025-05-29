#include <util/CudaUtil.hpp>

namespace cuda_util {

inline int get_device_id(const std::string& cuda_device) {
  auto pos = cuda_device.find(':');
  if (pos != std::string::npos) {
    return std::stoi(cuda_device.substr(pos + 1));
  } else {
    return std::stoi(cuda_device);
  }
}

inline void gpu2cpu_memcpy(void* dst, const void* src, size_t n_bytes) {
  cudaError_t err = cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw util::Exception("cudaMemcpy failed: {}", cudaGetErrorString(err));
  }
}

inline void* gpu_malloc(size_t n_bytes) {
  void* ptr;
  cudaError_t err = cudaMalloc(&ptr, n_bytes);
  if (err != cudaSuccess) {
    throw util::Exception("cudaMalloc failed: {}", cudaGetErrorString(err));
  }
  return ptr;
}

inline void assert_on_device(int device_id) {
  int current_device;
  cudaGetDevice(&current_device);
  if (current_device != device_id) {
    throw util::Exception("Expected CUDA device {}, found {}", device_id, current_device);
  }
}

inline size_t dim_size(const nvinfer1::Dims& dims) {
  size_t size = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];
  }
  return size;
}

}  // namespace cuda_util
