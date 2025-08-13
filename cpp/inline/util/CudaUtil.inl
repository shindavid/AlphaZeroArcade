#include "util/CudaUtil.hpp"

#include "util/Exceptions.hpp"

namespace cuda_util {

inline const char* get_sm_tag() {
  static std::string sm_tag;

  if (sm_tag.empty()) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);

    sm_tag = std::format("{}.{}", p.major, p.minor);
  }

  return sm_tag.c_str();
}

inline int cuda_device_to_ordinal(const std::string& cuda_device) {
  auto pos = cuda_device.find(':');
  if (pos != std::string::npos) {
    return std::stoi(cuda_device.substr(pos + 1));
  } else {
    return std::stoi(cuda_device);
  }
}

inline cudaStream_t create_stream() {
  cudaStream_t stream;
  cudaError_t err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    throw util::Exception("cudaStreamCreate failed: {}", cudaGetErrorString(err));
  }
  return stream;
}

inline void destroy_stream(cudaStream_t stream) {
  cudaError_t err = cudaStreamDestroy(stream);
  if (err != cudaSuccess) {
    throw util::Exception("cudaStreamDestroy failed: {}", cudaGetErrorString(err));
  }
}

inline void synchronize_stream(cudaStream_t stream) {
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    throw util::Exception("cudaStreamSynchronize failed: {}", cudaGetErrorString(err));
  }
}

inline void set_device(int device_id) {
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    // TODO: throw a CleanException if GPU is unavailable
    //
    // terminate called after throwing an instance of 'util::Exception'
    // what():  cudaSetDevice failed: no CUDA-capable device is detected
    throw util::Exception("cudaSetDevice failed: {}", cudaGetErrorString(err));
  }
}

inline void cpu2gpu_memcpy(void* dst, const void* src, size_t n_bytes) {
  cudaError_t err = cudaMemcpy(dst, src, n_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw util::Exception("cudaMemcpy failed: {}", cudaGetErrorString(err));
  }
}

inline void gpu2cpu_memcpy(void* dst, const void* src, size_t n_bytes) {
  cudaError_t err = cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw util::Exception("cudaMemcpy failed: {}", cudaGetErrorString(err));
  }
}

inline void cpu2gpu_memcpy_async(cudaStream_t stream, void* dst, const void* src, size_t n_bytes) {
  cudaError_t err = cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    throw util::Exception("cudaMemcpyAsync failed: {}", cudaGetErrorString(err));
  }
}

inline void gpu2cpu_memcpy_async(cudaStream_t stream, void* dst, const void* src, size_t n_bytes) {
  cudaError_t err = cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    throw util::Exception("cudaMemcpyAsync failed: {}", cudaGetErrorString(err));
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

inline void* cpu_malloc(size_t n_bytes) {
  void* ptr;
  cudaError_t err = cudaMallocHost(&ptr, n_bytes);
  if (err != cudaSuccess) {
    throw util::Exception("cudaMallocHost failed: {}", cudaGetErrorString(err));
  }
  return ptr;
}

inline void gpu_free(void* ptr) {
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    throw util::Exception("cudaFree failed: {}", cudaGetErrorString(err));
  }
}

inline void cpu_free(void* ptr) {
  cudaError_t err = cudaFreeHost(ptr);
  if (err != cudaSuccess) {
    throw util::Exception("cudaFreeHost failed: {}", cudaGetErrorString(err));
  }
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
