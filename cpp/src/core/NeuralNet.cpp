#include "core/NeuralNet.hpp"

#include "util/Exceptions.hpp"
#include "util/TensorRtUtil.hpp"

#include <chrono>
#include <memory>

namespace core {

NeuralNetBase::NeuralNetBase(const NeuralNetParams& params)
    : runtime_(nvinfer1::createInferRuntime(logger_)),
      params_(params) {}

NeuralNetBase::~NeuralNetBase() {
  delete parser_refitter_;
  delete refitter_;
  delete engine_;
  delete runtime_;
}

void NeuralNetBase::set_model_architecture_signature() {
  const char* key = "model-architecture-signature";
  onnx::ModelProto model;
  if (!model.ParseFromArray(onnx_bytes_.data(), (int)onnx_bytes_.size())) return;
  for (int i = 0; i < model.metadata_props_size(); ++i) {
    const auto& kv = model.metadata_props(i);
    if (kv.key() == key) {
      model_architecture_signature_ = kv.value();
      return;
    }
  }

  throw util::Exception("onnx model file missing metadata key {}", key);
}

void NeuralNetBase::load_data(std::vector<char>& dst, const char* filename) {
  std::ifstream f(filename, std::ios::binary | std::ios::ate);
  size_t sz = f.tellg();
  f.seekg(0);
  dst.resize(sz);
  f.read(dst.data(), sz);
}

void NeuralNetBase::load_data(std::vector<char>& dst, std::ispanstream& bytes) {
  dst.assign(std::istreambuf_iterator<char>(bytes), std::istreambuf_iterator<char>());
}

void NeuralNetBase::init_engine_from_plan_data() {
  engine_ = runtime_->deserializeCudaEngine(plan_data_.data(), plan_data_.size());
}

void NeuralNetBase::init_refitter() {
  if (parser_refitter_) return;

  refitter_ = nvinfer1::createInferRefitter(*engine_, logger_);
  parser_refitter_ = nvonnxparser::createParserRefitter(*refitter_, logger_);
}

void NeuralNetBase::refit_engine_plan() {
  // TODO: in the event of an exception, consider gracefully failing over to a fresh engine
  // build. We throw an exception for now because we want to understand if/why failures happen.

  init_refitter();

  auto t1 = std::chrono::steady_clock::now();
  if (!parser_refitter_->refitFromBytes(onnx_bytes_.data(), onnx_bytes_.size())) {
    throw util::Exception("Failed to refit parser from bytes: {}",
                          parser_refitter_->getError(0)->desc());
  }
  if (!refitter_->refitCudaEngine()) {
    throw util::Exception("Failed to refit CUDA engine");
  }
  auto t2 = std::chrono::steady_clock::now();
  auto refit_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG_INFO("Refit TensorRT engine in {} ms", refit_time_ms);
}

void NeuralNetBase::build_engine_plan_from_scratch() {
  LOG_INFO("Building a TensorRT engine from an ONNX model from scratch.");
  LOG_INFO("");
  LOG_INFO("** This will take a long time! **");
  LOG_INFO("");
  LOG_INFO(
    "However, future runs with this model architecture + precision + batch-size should avoid "
    "this one-time cost.");
  LOG_INFO("");

  auto t1 = std::chrono::steady_clock::now();
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
  std::unique_ptr<nvinfer1::INetworkDefinition> net_def(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net_def, logger_));
  if (!parser->parse(onnx_bytes_.data(), onnx_bytes_.size())) {
    throw util::Exception("Failed to parse ONNX model");
  }
  std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, params_.workspace_size_in_bytes);
  cfg->setFlag(trt_util::precision_to_builder_flag(params_.precision));
  cfg->setFlag(nvinfer1::BuilderFlag::kREFIT);
  auto* in = net_def->getInput(0);
  auto dims = in->getDimensions();

  nvinfer1::IOptimizationProfile* prof = builder->createOptimizationProfile();
  dims.d[0] = params_.batch_size;
  prof->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
  prof->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
  prof->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
  cfg->addOptimizationProfile(prof);

  auto t2 = std::chrono::steady_clock::now();
  engine_ = builder->buildEngineWithConfig(*net_def, *cfg);
  auto t3 = std::chrono::steady_clock::now();

  auto t12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  auto t23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  LOG_INFO("TensorRT Engine building prep-time: {} ms", t12);
  LOG_INFO("TensorRT Engine building time: {} ms", t23);
}

void NeuralNetBase::save_plan_bytes() {
  std::unique_ptr<nvinfer1::IHostMemory> mem(engine_->serialize());
  if (!mem) {
    throw util::Exception("Failed to serialize TensorRT engine");
  }
  plan_data_.assign((char*)mem->data(), (char*)mem->data() + mem->size());
}

void NeuralNetBase::write_plan_to_disk(const boost::filesystem::path& cache_path) {
  boost::filesystem::create_directories(cache_path.parent_path());

  std::string tmp = cache_path.string() + ".tmp";
  std::ofstream f(tmp, std::ios::binary);
  if (!f) {
    throw util::Exception("Failed to open temporary file {}", tmp);
  }
  f.write(reinterpret_cast<const char*>(plan_data_.data()), (std::streamsize)plan_data_.size());
  f.close();

  boost::filesystem::rename(tmp, cache_path);

  LOG_INFO("Successfully saved TensorRT engine plan to {}", cache_path.string());
}

}  // namespace core
