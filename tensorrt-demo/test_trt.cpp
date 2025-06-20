// test_trt.cpp

/*
Compilation cmd:

g++ test_trt.cpp -o test_trt \
  -I/usr/include/x86_64-linux-gnu \
  -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lcudart \
  -std=c++17 -O2
*/
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace nvinfer1;

// simple logger
class Logger : public ILogger {
 public:
  void log(Severity severity, AsciiChar const* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " engine.plan\n";
    return 1;
  }

  // 1) read plan file
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error opening engine file\n";
    return 1;
  }
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engineData(size);
  file.read(engineData.data(), size);
  file.close();

  // 2) deserialize
  Logger logger;
  IRuntime* runtime = createInferRuntime(logger);
  if (!runtime) {
    std::cerr << "createInferRuntime failed\n";
    return 1;
  }
  ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
  if (!engine) {
    std::cerr << "deserialize failed\n";
    delete runtime;
    return 1;
  }

  // 3) execution context
  IExecutionContext* ctx = engine->createExecutionContext();
  if (!ctx) {
    std::cerr << "createExecutionContext failed\n";
    delete engine;
    delete runtime;
    return 1;
  }

  // 4) get I/O names & shapes
  int nbIO = engine->getNbIOTensors();
  if (nbIO < 2) {
    std::cerr << "Expected at least 2 I/O tensors, got " << nbIO << "\n";
    delete ctx;
    delete engine;
    delete runtime;
    return 1;
  }

  const char* inputName = engine->getIOTensorName(0);
  const char* outputName = engine->getIOTensorName(1);
  // Dims inDims = engine->getTensorShape(inputName);
  // Dims outDims = engine->getTensorShape(outputName);

  // 4a) pick profile and set input shape
  ctx->setOptimizationProfileAsync(0, 0);
  Dims inOpt  = engine->getProfileShape(inputName,  0, OptProfileSelector::kOPT);
  Dims outOpt = engine->getProfileShape(outputName, 0, OptProfileSelector::kOPT);

  if (!ctx->setInputShape(inputName, inOpt)) {
    std::cerr << "Bad input shape\n";
    return 1;
  }

  // now inOpt/outOpt are fully concrete (no -1’s) so we can size our buffers:
  int64_t inCount = 1;
  for (int i = 0; i < inOpt.nbDims; ++i) inCount  *= inOpt.d[i];
  int64_t outCount = 1;
  for (int i = 0; i < outOpt.nbDims; ++i) outCount *= outOpt.d[i];

  // 5) host/device buffers
  std::vector<float> hostIn(inCount, 0.0f), hostOut(outCount);
  std::vector<void*> deviceBuffers(nbIO, nullptr);
  cudaMalloc(&deviceBuffers[0], inCount  * sizeof(float));
  cudaMalloc(&deviceBuffers[1], outCount * sizeof(float));
  cudaMemcpy(deviceBuffers[0], hostIn.data(), inCount * sizeof(float), cudaMemcpyHostToDevice);

  // 6) inference
  if (!ctx->executeV2(deviceBuffers.data())) {
    std::cerr << "Inference failed\n";
  } else {
    cudaMemcpy(hostOut.data(), deviceBuffers[1], outCount * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Output: " << hostOut[0] << " " << hostOut[1] << "\n";
  }

  // 7) cleanup
  cudaFree(deviceBuffers[0]);
  cudaFree(deviceBuffers[1]);
  delete ctx;
  delete engine;
  delete runtime;

  return 0;
}
