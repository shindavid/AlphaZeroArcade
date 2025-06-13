/*
 * Built by build.py, called from python to build an onnx plan file from an ONNX model.
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
  // min = maxBatch
  dims.d[0] = maxBatch;
  profile->setDimensions(network->getInput(0)->getName(),
                         OptProfileSelector::kMIN, dims);
  // opt = maxBatch
  dims.d[0] = maxBatch;
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

  delete plan;
  delete engine;
  delete config;
  delete parser;
  delete network;
  delete builder;
  return 0;
}
