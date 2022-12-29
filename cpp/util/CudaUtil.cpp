#include <util/CudaUtil.hpp>

#include <cstdint>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cuda {

void dump_memory_info() {
  int num_gpus;
  size_t free, total;
  cudaGetDeviceCount( &num_gpus );
  for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
    cudaSetDevice( gpu_id );
    int id;
    cudaGetDevice( &id );
    cudaMemGetInfo( &free, &total );
    std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
  }
}

}  // namespace cuda
