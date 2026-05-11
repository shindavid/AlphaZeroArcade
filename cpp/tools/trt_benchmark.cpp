#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <boost/program_options.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

struct PerfStats {
    float batch_per_second = -1.0f;
    float items_per_second = -1.0f;
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

ICudaEngine* buildEngine(const std::string& onnxFile, int minBatch, int optBatch, int maxBatch, const std::string& precision, const ModelConfig& config) {
    std::cout << "Building engine with MIN=" << minBatch
              << ", OPT=" << optBatch << ", MAX=" << maxBatch
              << ", PRECISION=" << precision << " ... " << std::flush;

    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING));

    IBuilderConfig* builderConfig = builder->createBuilderConfig();

    if (precision == "fp16") {
        builderConfig->setFlag(BuilderFlag::kFP16);
    } else if (precision == "int8") {
        builderConfig->setFlag(BuilderFlag::kINT8);
    }  // "fp32" is the default behavior, so no flag is needed

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

void printTable(const std::string& title,
                const std::vector<PerfStats>& results,
                int minBatch,
                const std::vector<int>& testBatchSizes,
                std::function<float(const PerfStats&)> extractor) {

    std::cout << "\n--- " << title << " ---\n";
    std::cout << "MIN  |  actual batch size\n";
    std::cout << "     |";
    for (int bs : testBatchSizes) {
        std::cout << std::setw(8) << bs;
    }
    std::cout << "\n----------------------------------------------------------------------\n";

    std::cout << std::setw(4) << minBatch << " |";
    for (size_t j = 0; j < testBatchSizes.size(); ++j) {
        float val = extractor(results[j]);
        if (val < 0.0f) {
            std::cout << std::setw(8) << "N/A";
        } else {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << val;
        }
    }
    std::cout << "\n";
}

struct Args {
    std::string model_path;
    std::string precision = "fp16";
    int min_batch = 1;
    int max_batch = 4;
    int opt_batch = 4;
    int warmup_runs = 10;
    int num_runs = 100;
    std::vector<int> test_batch_sizes;

    void populate_test_batch_sizes() {
      for (int i = 0; (1 << i) <= max_batch; ++i) {
        test_batch_sizes.push_back(1 << i);
      }
    }
};

bool parseArgs(int argc, char** argv, Args& args) {
    namespace po = boost::program_options;

    po::options_description desc("trt_benchmark options");
    desc.add_options()
        ("help", "show help message")
        ("model", po::value<std::string>(&args.model_path), "path to model .onnx")
        ("precision", po::value<std::string>(&args.precision)->default_value(args.precision), "Build precision: fp32, fp16, or int8")
        ("min-batch", po::value<int>(&args.min_batch)->default_value(args.min_batch), "min batch size in optimization profile")
        ("max-batch", po::value<int>(&args.max_batch)->default_value(args.max_batch), "max batch size in optimization profile")
        ("opt-batch", po::value<int>(&args.opt_batch)->default_value(args.opt_batch), "opt batch size in optimization profile")
        ("warmup-runs", po::value<int>(&args.warmup_runs)->default_value(args.warmup_runs), "number of warmup inferences")
        ("num-runs", po::value<int>(&args.num_runs)->default_value(args.num_runs), "number of timed inferences");

    po::positional_options_description positional;
    positional.add("model", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
        po::notify(vm);
        args.populate_test_batch_sizes();
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n\n";
        std::cerr << desc << std::endl;
        return false;
    }

    if (vm.contains("help")) {
        std::cout << desc << std::endl;
        return false;
    }

    // Normalize string to lowercase and validate
    std::transform(args.precision.begin(), args.precision.end(), args.precision.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (args.precision != "fp32" && args.precision != "fp16" && args.precision != "int8") {
        std::cerr << "Invalid precision '" << args.precision << "'. Allowed values: fp32, fp16, int8.\n\n";
        std::cerr << desc << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        return 1;
    }

    ModelConfig config;
    if (!extractModelConfig(args.model_path, config)) return -1;

    std::vector<void*> d_inputs(config.inputs.size(), nullptr);
    for (size_t i = 0; i < config.inputs.size(); ++i) {
        CHECK_CUDA(cudaMalloc(&d_inputs[i], args.max_batch * config.inputs[i].volPerBatch * sizeof(float)));
    }

    std::vector<void*> d_outputs(config.outputs.size(), nullptr);
    for (size_t i = 0; i < config.outputs.size(); ++i) {
        CHECK_CUDA(cudaMalloc(&d_outputs[i], args.max_batch * config.outputs[i].volPerBatch * sizeof(float)));
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::vector<PerfStats> results(args.test_batch_sizes.size());

    std::cout << "\nStarting Benchmark (" << args.num_runs << " runs per test)..." << std::endl;

    ICudaEngine* engine = buildEngine(args.model_path, args.min_batch, args.opt_batch,
                                      args.max_batch, args.precision, config);
    if (engine) {
        IExecutionContext* context = engine->createExecutionContext();

        for (size_t j = 0; j < args.test_batch_sizes.size(); ++j) {
            int actualBatch = args.test_batch_sizes[j];
            if (actualBatch < args.min_batch) continue;

            results[j] = timeInferenceV3(
                context, stream, config, d_inputs, d_outputs, actualBatch, args.warmup_runs, args.num_runs);
        }

        delete context;
        delete engine;
    }

    printTable("Batch per Second", results, args.min_batch, args.test_batch_sizes,
               [](const PerfStats& s) { return s.batch_per_second; });
    printTable("Items per Second", results, args.min_batch, args.test_batch_sizes,
               [](const PerfStats& s) { return s.items_per_second; });

    for (void* ptr : d_inputs) CHECK_CUDA(cudaFree(ptr));
    for (void* ptr : d_outputs) CHECK_CUDA(cudaFree(ptr));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
