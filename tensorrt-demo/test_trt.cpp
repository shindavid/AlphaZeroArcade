// test_trt.cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

using namespace nvinfer1;

// Simple logger for TRT
class Logger : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cout << msg << "\n";
  }
} gLogger;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " model.onnx\n";
    return 1;
  }
  const char* onnx = argv[1];

  // 1) Builder + network + parser
  auto builder = createInferBuilder(gLogger);

  // implicit-batch mode: no flags needed
  auto network = builder->createNetworkV2(0);
  auto parser  = nvonnxparser::createParser(*network, gLogger);

  if (!parser->parseFromFile(onnx, (int)ILogger::Severity::kWARNING)) {
    std::cerr << "Failed to parse ONNX\n";
    return 1;
  }

  // 2) Build engine
  auto config = builder->createBuilderConfig();
  // set workspace limit (1 MiB here; TRT will round as needed)
  config->setMemoryPoolLimit(
    nvinfer1::MemoryPoolType::kWORKSPACE,
    1 << 20
  );
  auto engine = builder->buildEngineWithConfig(*network, *config);

  // 3) Create context
  auto ctx = engine->createExecutionContext();

  // 4) Hard-coded dims for our tiny test (batch=1, in=3, out=2)
  const int B = 1, IN = 3, OUT = 2;
  const size_t inSize  = B * IN;
  const size_t outSize = B * OUT;

  // 5) Allocate host & device buffers
  std::vector<float> hostIn(inSize, 1.0f), hostOut(outSize);
  void* buffers[2] = {};
  cudaMalloc(&buffers[0], inSize  * sizeof(float));
  cudaMalloc(&buffers[1], outSize * sizeof(float));

  // 6) Copy input → GPU
  cudaMemcpy(buffers[0], hostIn.data(),
             inSize * sizeof(float),
             cudaMemcpyHostToDevice);

  // 7) Inference (binding 0=input, 1=output)
  ctx->executeV2(buffers);

  // 8) Copy GPU → host
  cudaMemcpy(hostOut.data(), buffers[1],
             outSize * sizeof(float),
             cudaMemcpyDeviceToHost);

  // 9) Print results
  std::cout << "Output:";
  for (float v : hostOut) std::cout << " " << v;
  std::cout << "\n";

  // 10) Cleanup
  cudaFree(buffers[0]);
  cudaFree(buffers[1]);
  return 0;
}
