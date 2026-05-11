#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << "[TRT ERROR] " << msg << std::endl;
        }
    }
} gLogger;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at line " << __LINE__ << ": " \
                      << cudaGetErrorString(status) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Struct for individual tensors
struct TensorConfig {
    std::string name;
    Dims dims;
    size_t volPerBatch;
};

// Struct to hold all inputs and outputs
struct ModelConfig {
    std::vector<TensorConfig> inputs;
    std::vector<TensorConfig> outputs;
};

size_t getVolumePerBatch(const Dims& d) {
    size_t vol = 1;
    for (int i = 1; i < d.nbDims; ++i) {
        if (d.d[i] > 0) vol *= d.d[i];
    }
    return vol;
}

// Extract configuration for ALL inputs and outputs
bool extractModelConfig(const std::string& onnxFile, ModelConfig& config) {
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file to extract config!" << std::endl;
        return false;
    }

    std::cout << "Detected Inputs:\n";
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto t = network->getInput(i);
        config.inputs.push_back({t->getName(), t->getDimensions(), getVolumePerBatch(t->getDimensions())});
        std::cout << "  - " << t->getName() << "\n";
    }

    std::cout << "Detected Outputs:\n";
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto t = network->getOutput(i);
        config.outputs.push_back({t->getName(), t->getDimensions(), getVolumePerBatch(t->getDimensions())});
        std::cout << "  - " << t->getName() << "\n";
    }

    delete parser;
    delete network;
    delete builder;
    return true;
}

ICudaEngine* buildEngine(const std::string& onnxFile, int minBatch, int optBatch, int maxBatch, const ModelConfig& config) {
    std::cout << "Building engine with MIN=" << minBatch
              << ", OPT=" << optBatch << ", MAX=" << maxBatch << " ... " << std::flush;

    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING));

    IBuilderConfig* builderConfig = builder->createBuilderConfig();
    IOptimizationProfile* profile = builder->createOptimizationProfile();

    // Apply dynamic batch profiles to ALL inputs
    for (const auto& in : config.inputs) {
        Dims minDims = in.dims; minDims.d[0] = minBatch;
        Dims optDims = in.dims; optDims.d[0] = optBatch;
        Dims maxDims = in.dims; maxDims.d[0] = maxBatch;

        profile->setDimensions(in.name.c_str(), OptProfileSelector::kMIN, minDims);
        profile->setDimensions(in.name.c_str(), OptProfileSelector::kOPT, optDims);
        profile->setDimensions(in.name.c_str(), OptProfileSelector::kMAX, maxDims);
    }

    builderConfig->addOptimizationProfile(profile);

    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *builderConfig);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

    delete serializedModel;
    delete parser;
    delete network;
    delete builderConfig;
    delete builder;
    delete runtime;

    std::cout << "Done." << std::endl;
    return engine;
}

float timeInferenceV3(IExecutionContext* context, cudaStream_t stream,
                      const ModelConfig& config,
                      const std::vector<void*>& d_inputs,
                      const std::vector<void*>& d_outputs,
                      int actualBatch, int warmupRuns = 10, int numRuns = 100) {

    // 1. Set runtime dimensions & bind addresses for ALL inputs
    for (size_t i = 0; i < config.inputs.size(); ++i) {
        Dims actualShape = config.inputs[i].dims;
        actualShape.d[0] = actualBatch;
        context->setInputShape(config.inputs[i].name.c_str(), actualShape);
        context->setTensorAddress(config.inputs[i].name.c_str(), d_inputs[i]);
    }

    // 2. Bind addresses for ALL outputs
    for (size_t i = 0; i < config.outputs.size(); ++i) {
        context->setTensorAddress(config.outputs[i].name.c_str(), d_outputs[i]);
    }

    // Warmup
    for (int i = 0; i < warmupRuns; ++i) {
        context->enqueueV3(stream);
    }
    cudaStreamSynchronize(stream);

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numRuns; ++i) {
        context->enqueueV3(stream);
    }
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count() / numRuns;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx>" << std::endl;
        return -1;
    }

    std::string MODEL_PATH = argv[1];
    const int MAX_BATCH = 64;
    const std::vector<int> minBatchScenarios = {1, MAX_BATCH};
    const std::vector<int> testBatchSizes = {1, 2, 4, 8, 16, 32, 64};

    // 1. Extract dynamic configuration
    ModelConfig config;
    if (!extractModelConfig(MODEL_PATH, config)) {
        return -1;
    }

    // 2. Allocate memory for ALL inputs and outputs
    std::vector<void*> d_inputs(config.inputs.size(), nullptr);
    for (size_t i = 0; i < config.inputs.size(); ++i) {
        size_t volBytes = MAX_BATCH * config.inputs[i].volPerBatch * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_inputs[i], volBytes));
    }

    std::vector<void*> d_outputs(config.outputs.size(), nullptr);
    for (size_t i = 0; i < config.outputs.size(); ++i) {
        size_t volBytes = MAX_BATCH * config.outputs[i].volPerBatch * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_outputs[i], volBytes));
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::vector<std::vector<float>> results(minBatchScenarios.size(), std::vector<float>(testBatchSizes.size(), 0.0f));
    std::cout << "\nStarting Benchmark..." << std::endl;

    for (size_t i = 0; i < minBatchScenarios.size(); ++i) {
        int minBatch = minBatchScenarios[i];
        ICudaEngine* engine = buildEngine(MODEL_PATH, minBatch, MAX_BATCH, MAX_BATCH, config);
        if (!engine) continue;

        IExecutionContext* context = engine->createExecutionContext();

        for (size_t j = 0; j < testBatchSizes.size(); ++j) {
            int actualBatch = testBatchSizes[j];
            if (actualBatch < minBatch) {
                results[i][j] = -1.0f;
                continue;
            }

            results[i][j] = timeInferenceV3(context, stream, config, d_inputs, d_outputs, actualBatch);
        }

        delete context;
        delete engine;
    }

    // 3. Print Matrix
    std::cout << "\nBatch Size Matrix (Inference Latency in ms)\n";
    std::cout << "MIN  |  actual batch size\n";
    std::cout << "     |";
    for (int bs : testBatchSizes) {
        std::cout << std::setw(8) << bs;
    }
    std::cout << "\n----------------------------------------------------------------------\n";

    for (size_t i = 0; i < minBatchScenarios.size(); ++i) {
        std::cout << std::setw(4) << minBatchScenarios[i] << " |";
        for (size_t j = 0; j < testBatchSizes.size(); ++j) {
            if (results[i][j] < 0.0f) {
                std::cout << std::setw(8) << "N/A";
            } else {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << results[i][j];
            }
        }
        std::cout << "\n";
    }

    // Cleanup
    for (void* ptr : d_inputs) CHECK_CUDA(cudaFree(ptr));
    for (void* ptr : d_outputs) CHECK_CUDA(cudaFree(ptr));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
