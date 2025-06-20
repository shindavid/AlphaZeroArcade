#include <util/CudaUtil.hpp>

#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cuda {

void dump_memory_info() {
  int num_gpus;
  size_t free, total;
  cudaGetDeviceCount(&num_gpus);
  for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    cudaSetDevice(gpu_id);
    int id;
    cudaGetDevice(&id);
    cudaMemGetInfo(&free, &total);
    std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
  }
}

void validate_batch_size(nvinfer1::ICudaEngine* engine, const char* tensor_name,
                                int profile_index, int batch_size) {
  using namespace nvinfer1;
  auto min_dims = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kMIN);
  auto max_dims = engine->getProfileShape(tensor_name, profile_index, OptProfileSelector::kMAX);

  int min0 = min_dims.d[0];
  int max0 = max_dims.d[0];

  // 1) Are we really dynamic?
  if (min0 == max0) {
    throw util::Exception("Engine for '{}' is built with a *fixed* batch size of {}",
                          tensor_name, min0);
  }

  // 2) Does our request fit under the hood?
  if (batch_size < min0 || batch_size > max0) {
    throw util::Exception(
      "Request batch size of {} for '{}' is outside of the engine’s supported range [{}, {}]",
      batch_size, tensor_name, min0, max0);
  }
}

}  // namespace cuda
