add_executable(trt_benchmark trt_benchmark.cpp)
target_include_directories(trt_benchmark PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_benchmark
  PRIVATE
  Boost::program_options
  CUDA::cudart
  ${TENSORRT_LIBS}
)
append_target_metadata("trt_benchmark")
