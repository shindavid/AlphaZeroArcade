#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <algorithm>
#include <functional>
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

// ---------------------------------------------------------
// NEW: Struct to hold multiple performance metrics
// ---------------------------------------------------------
struct PerfStats {
    float batch_per_second;
    float items_per_second;
};

struct TensorConfig {
    std::string name;
    Dims dims;
    size_t volPerBatch;
};

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

bool extractModelConfig(const std::string& onnxFile, ModelConfig& config) {
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file to extract config!" << std::endl;
        return false;
    }

    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto t = network->getInput(i);
        config.inputs.push_back({t->getName(), t->getDimensions(), getVolumePerBatch(t->getDimensions())});
    }

    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto t = network->getOutput(i);
        config.outputs.push_back({t->getName(), t->getDimensions(), getVolumePerBatch(t->getDimensions())});
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
    builderConfig->setFlag(BuilderFlag::kFP16);

    IOptimizationProfile* profile = builder->createOptimizationProfile();

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

// ---------------------------------------------------------
// UPDATED: Now returns PerfStats and times per-inference
// ---------------------------------------------------------
PerfStats timeInferenceV3(IExecutionContext* context, cudaStream_t stream,
                          const ModelConfig& config,
                          const std::vector<void*>& d_inputs,
                          const std::vector<void*>& d_outputs,
                          int actualBatch, int warmupRuns = 10, int numRuns = 100) {

    for (size_t i = 0; i < config.inputs.size(); ++i) {
        Dims actualShape = config.inputs[i].dims;
        actualShape.d[0] = actualBatch;
        context->setInputShape(config.inputs[i].name.c_str(), actualShape);
        context->setTensorAddress(config.inputs[i].name.c_str(), d_inputs[i]);
    }

    for (size_t i = 0; i < config.outputs.size(); ++i) {
        context->setTensorAddress(config.outputs[i].name.c_str(), d_outputs[i]);
    }

    // Warmup
    for (int i = 0; i < warmupRuns; ++i) {
        context->enqueueV3(stream);
    }
    cudaStreamSynchronize(stream);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numRuns; ++i) {
        context->enqueueV3(stream);
    }
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> total_seconds = end - start;
    float batch_per_second = numRuns / total_seconds.count();
    float items_per_second = (numRuns * actualBatch) / total_seconds.count();

    PerfStats stats;
    stats.batch_per_second = batch_per_second;
    stats.items_per_second = items_per_second;

    return stats;
}

// ---------------------------------------------------------
// NEW: Helper function to cleanly print different stats
// ---------------------------------------------------------
void printTable(const std::string& title,
                const std::vector<std::vector<PerfStats>>& results,
                const std::vector<int>& minBatchScenarios,
                const std::vector<int>& testBatchSizes,
                std::function<float(const PerfStats&)> extractor) {

    std::cout << "\n--- " << title << " (ms) ---\n";
    std::cout << "MIN  |  actual batch size\n";
    std::cout << "     |";
    for (int bs : testBatchSizes) {
        std::cout << std::setw(8) << bs;
    }
    std::cout << "\n----------------------------------------------------------------------\n";

    for (size_t i = 0; i < minBatchScenarios.size(); ++i) {
        std::cout << std::setw(4) << minBatchScenarios[i] << " |";
        for (size_t j = 0; j < testBatchSizes.size(); ++j) {
            float val = extractor(results[i][j]);
            if (val < 0.0f) {
                std::cout << std::setw(8) << "N/A";
            } else {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << val;
            }
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx>" << std::endl;
        return -1;
    }

    std::string MODEL_PATH = argv[1];
    const int MAX_BATCH = 4;
    const int OPT_BATCH = 4;
    const std::vector<int> minBatchScenarios = {1};
    const std::vector<int> testBatchSizes = {1, 2, 4};

    ModelConfig config;
    if (!extractModelConfig(MODEL_PATH, config)) return -1;

    std::vector<void*> d_inputs(config.inputs.size(), nullptr);
    for (size_t i = 0; i < config.inputs.size(); ++i) {
        CHECK_CUDA(cudaMalloc(&d_inputs[i], MAX_BATCH * config.inputs[i].volPerBatch * sizeof(float)));
    }

    std::vector<void*> d_outputs(config.outputs.size(), nullptr);
    for (size_t i = 0; i < config.outputs.size(); ++i) {
        CHECK_CUDA(cudaMalloc(&d_outputs[i], MAX_BATCH * config.outputs[i].volPerBatch * sizeof(float)));
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Matrix now stores PerfStats instead of a single float
    std::vector<std::vector<PerfStats>> results(minBatchScenarios.size(), std::vector<PerfStats>(testBatchSizes.size()));

    std::cout << "\nStarting Benchmark (100 runs per test)..." << std::endl;

    for (size_t i = 0; i < minBatchScenarios.size(); ++i) {
        int minBatch = minBatchScenarios[i];
        ICudaEngine* engine = buildEngine(MODEL_PATH, minBatch, OPT_BATCH, MAX_BATCH, config);
        if (!engine) continue;

        IExecutionContext* context = engine->createExecutionContext();

        for (size_t j = 0; j < testBatchSizes.size(); ++j) {
            int actualBatch = testBatchSizes[j];
            if (actualBatch < minBatch) continue; // Leaves defaults at -1.0f

            // Increased default runs to 100 for better percentile accuracy
            results[i][j] = timeInferenceV3(context, stream, config, d_inputs, d_outputs, actualBatch, 10, 100);
        }

        delete context;
        delete engine;
    }

    // ---------------------------------------------------------
    // Output Multiple Tables!
    // ---------------------------------------------------------
    printTable("Batch per Second", results, minBatchScenarios, testBatchSizes, [](const PerfStats& s){ return s.batch_per_second; });
    printTable("Items per Second", results, minBatchScenarios, testBatchSizes, [](const PerfStats& s){ return s.items_per_second; });

    // Cleanup
    for (void* ptr : d_inputs) CHECK_CUDA(cudaFree(ptr));
    for (void* ptr : d_outputs) CHECK_CUDA(cudaFree(ptr));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
