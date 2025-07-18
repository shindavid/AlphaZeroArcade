// build_engine.cpp

/*
Compilation cmd:

g++ build_engine.cpp -o build_engine \
  -I/usr/include/x86_64-linux-gnu -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64 \
  -lnvinfer -lnvonnxparser -lnvinfer_plugin -lcudart \
  -std=c++17 -O2

*/
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

// Simple logger that prints warnings & errors
class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << msg << std::endl;
    }
  }
} gLogger;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " model.onnx engine.plan maxBatch\n";
    return 1;
  }
  const char* onnxFile = argv[1];
  const char* planFile = argv[2];
  int maxBatch = std::atoi(argv[3]);

  // 1) Builder + network + parser
  auto builder = createInferBuilder(gLogger);
  auto network = builder->createNetworkV2(0);
  auto parser  = nvonnxparser::createParser(*network, gLogger);
  if (!parser->parseFromFile(onnxFile,
        static_cast<int>(ILogger::Severity::kWARNING))) {
    std::cerr << "ERROR: failed to parse ONNX model\n";
    return 1;
  }

  // 2) Create builder config & optimization profile
  auto config = builder->createBuilderConfig();
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);
  auto profile = builder->createOptimizationProfile();
  Dims dims = network->getInput(0)->getDimensions();
  // min = 1
  dims.d[0] = 1;
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kMIN, dims);
  // opt = 1
  dims.d[0] = 1;
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kOPT, dims);
  // max = maxBatch
  dims.d[0] = maxBatch;
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kMAX, dims);
  config->addOptimizationProfile(profile);

  // 3) Build and serialize engine
  auto engine = builder->buildEngineWithConfig(*network, *config);
  if (!engine) {
    std::cerr << "ERROR: engine build failed\n";
    return 1;
  }
  auto plan = engine->serialize();
  std::ofstream out(planFile, std::ios::binary);
  out.write(reinterpret_cast<const char*>(plan->data()), plan->size());
  std::cout << "Engine built and serialized to " << planFile << "\n";
  return 0;
}
