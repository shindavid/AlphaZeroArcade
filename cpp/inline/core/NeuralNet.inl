#include "core/NeuralNet.hpp"

#include "util/Asserts.hpp"
#include "util/CudaUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/TensorRtUtil.hpp"

#include <onnx/onnx_pb.h>

#include <NvInfer.h>
#include <memory>

namespace core {

namespace detail {

template <eigen_util::concepts::Shape Shape>
float* make_ptr(int batch_size) {
  size_t mult = batch_size * sizeof(float);
  return (float*)cuda_util::cpu_malloc(mult * Shape::TotalSize());
}

template <eigen_util::concepts::Shape Shape>
auto make_arr(int batch_size) {
  return util::to_std_array<int64_t>(batch_size, eigen_util::to_int64_std_array_v<Shape>);
}

}  // namespace detail

template <concepts::Game Game>
NeuralNet<Game>::NeuralNet(const NeuralNetParams& params)
    : runtime_(nvinfer1::createInferRuntime(logger_)),
      params_(params) {}

template <concepts::Game Game>
NeuralNet<Game>::~NeuralNet() {
  deactivate();

  for (Pipeline* pipeline : pipelines_) {
    delete pipeline;
  }
  delete parser_refitter_;
  delete refitter_;
  delete engine_;
  delete runtime_;
}

template <concepts::Game Game>
template <typename T>
void NeuralNet<Game>::load_weights(T&& onnx_data) {
  cuda_util::set_device(params_.cuda_device_id);
  load_data(onnx_bytes_, std::forward<T>(onnx_data));

  std::string cur_signature = model_architecture_signature_;
  set_model_architecture_signature();
  boost::filesystem::path cache_path =
    trt_util::get_engine_plan_cache_path(model_architecture_signature_, params_.precision,
                                         params_.workspace_size_in_bytes, params_.batch_size);

  bool refit = cur_signature == model_architecture_signature_;
  if (!refit) {
    // This indicates that we don't have an existing model already loaded with a matching model
    // architecture signature (MAS). Let's look in the filesystem cache for a compatible model.
    if (boost::filesystem::exists(cache_path)) {
      LOG_INFO("Found cached engine plan at {}", cache_path.string());
      load_data(plan_data_, cache_path.string().c_str());
      engine_ = runtime_->deserializeCudaEngine(plan_data_.data(), plan_data_.size());
      refit = true;
    }
  }

  if (refit) {
    refit_engine_plan();
    save_plan_bytes();
  } else {
    build_engine_plan_from_scratch();
    save_plan_bytes();
    write_plan_to_disk(cache_path);
  }
}

template <concepts::Game Game>
pipeline_index_t NeuralNet<Game>::get_pipeline_assignment() {
  mit::unique_lock lock(pipeline_mutex_);
  pipeline_cv_.wait(lock, [&] { return !available_pipeline_indices_.empty(); });
  pipeline_index_t index = available_pipeline_indices_.back();
  available_pipeline_indices_.pop_back();
  return index;
}

template <concepts::Game Game>
float* NeuralNet<Game>::get_input_ptr(pipeline_index_t index) {
  return pipelines_[index]->input.data();
}

template <concepts::Game Game>
void NeuralNet<Game>::schedule(pipeline_index_t index) const {
  RELEASE_ASSERT(activated(), "NeuralNet<Game>::predict() called while deactivated");
  pipelines_[index]->schedule();
}

template <concepts::Game Game>
void NeuralNet<Game>::release(pipeline_index_t index) {
  mit::unique_lock lock(pipeline_mutex_);
  available_pipeline_indices_.push_back(index);
  lock.unlock();
  pipeline_cv_.notify_all();
}

template <concepts::Game Game>
void NeuralNet<Game>::load(pipeline_index_t index, float** policy_data, float** value_data,
                           float** action_values_data) {
  pipelines_[index]->load(policy_data, value_data, action_values_data);
}

template <concepts::Game Game>
void NeuralNet<Game>::deactivate() {
  if (!activated()) return;

  LOG_DEBUG("Deactivating NeuralNet...");

  activated_ = false;
  for (Pipeline* pipeline : pipelines_) {
    delete pipeline;
  }
  pipelines_.clear();
  {
    mit::unique_lock lock(pipeline_mutex_);
    available_pipeline_indices_.clear();
  }

  delete engine_;
  engine_ = nullptr;
}

template <concepts::Game Game>
bool NeuralNet<Game>::activate(int num_pipelines) {
  if (activated()) return false;

  LOG_DEBUG("Activating NeuralNet ({})...", num_pipelines);

  activated_ = true;
  RELEASE_ASSERT(loaded(), "NeuralNet<Game>::{}() called before weights loaded", __func__);

  cuda_util::set_device(params_.cuda_device_id);
  engine_ = runtime_->deserializeCudaEngine(plan_data_.data(), plan_data_.size());

  nvinfer1::Dims input_shape =
    engine_->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kOPT);

  RELEASE_ASSERT(pipelines_.empty());

  {
    mit::unique_lock lock(pipeline_mutex_);
    RELEASE_ASSERT(available_pipeline_indices_.empty());
    for (int i = 0; i < num_pipelines; ++i) {
      pipelines_.push_back(new Pipeline(engine_, input_shape, params_.batch_size));
      available_pipeline_indices_.push_back(i);
    }
  }
  pipeline_cv_.notify_all();

  LOG_DEBUG("Done activating NeuralNet ({})!", num_pipelines);
  return true;
}

template <concepts::Game Game>
NeuralNet<Game>::Pipeline::Pipeline(nvinfer1::ICudaEngine* engine,
                                    const nvinfer1::Dims& input_shape, int batch_size)
    : input(detail::make_ptr<InputShape>(batch_size), detail::make_arr<InputShape>(batch_size)),
      policy(detail::make_ptr<PolicyShape>(batch_size), detail::make_arr<PolicyShape>(batch_size)),
      value(detail::make_ptr<ValueShape>(batch_size), detail::make_arr<ValueShape>(batch_size)),
      action_values(detail::make_ptr<ActionValueShape>(batch_size),
                    detail::make_arr<ActionValueShape>(batch_size)) {
  constexpr size_t f = sizeof(float);
  device_buffers.push_back(cuda_util::gpu_malloc(f * input.size()));
  device_buffers.push_back(cuda_util::gpu_malloc(f * policy.size()));
  device_buffers.push_back(cuda_util::gpu_malloc(f * value.size()));
  device_buffers.push_back(cuda_util::gpu_malloc(f * action_values.size()));

  context = engine->createExecutionContext();
  for (int i = 0; i < (int)device_buffers.size(); ++i) {
    context->setTensorAddress(engine->getIOTensorName(i), device_buffers[i]);
  }
  stream = cuda_util::create_stream();
  context->setOptimizationProfileAsync(0, stream);

  if (!context->setInputShape("input", input_shape)) throw std::runtime_error("bad input shape");
}

template <concepts::Game Game>
NeuralNet<Game>::Pipeline::~Pipeline() {
  cuda_util::destroy_stream(stream);
  delete context;

  for (void* ptr : device_buffers) {
    cuda_util::gpu_free(ptr);
  }
  device_buffers.clear();

  cuda_util::cpu_free(input.data());
  cuda_util::cpu_free(policy.data());
  cuda_util::cpu_free(value.data());
  cuda_util::cpu_free(action_values.data());
}

template <concepts::Game Game>
void NeuralNet<Game>::Pipeline::schedule() {
  constexpr size_t f = sizeof(float);
  auto& dbs = device_buffers;
  cuda_util::cpu2gpu_memcpy_async(stream, dbs[0], input.data(), input.size() * f);

  bool ok = context->enqueueV3(stream);
  if (!ok) throw std::runtime_error("TensorRT inference failed");

  cuda_util::gpu2cpu_memcpy_async(stream, policy.data(), dbs[1], policy.size() * f);
  cuda_util::gpu2cpu_memcpy_async(stream, value.data(), dbs[2], value.size() * f);
  cuda_util::gpu2cpu_memcpy_async(stream, action_values.data(), dbs[3], action_values.size() * f);
}

template <concepts::Game Game>
void NeuralNet<Game>::Pipeline::load(float** policy_data, float** value_data,
                                     float** action_values_data) {
  cuda_util::synchronize_stream(stream);
  *policy_data = policy.data();
  *value_data = value.data();
  *action_values_data = action_values.data();
}

template <concepts::Game Game>
void NeuralNet<Game>::init_refitter() {
  if (parser_refitter_) return;

  refitter_ = nvinfer1::createInferRefitter(*engine_, logger_);
  parser_refitter_ = nvonnxparser::createParserRefitter(*refitter_, logger_);
}

template <concepts::Game Game>
void NeuralNet<Game>::refit_engine_plan() {
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

template <concepts::Game Game>
void NeuralNet<Game>::build_engine_plan_from_scratch() {
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

template <concepts::Game Game>
void NeuralNet<Game>::save_plan_bytes() {
  std::unique_ptr<nvinfer1::IHostMemory> mem(engine_->serialize());
  if (!mem) {
    throw util::Exception("Failed to serialize TensorRT engine");
  }
  plan_data_.assign((char*)mem->data(), (char*)mem->data() + mem->size());
}

template <concepts::Game Game>
void NeuralNet<Game>::write_plan_to_disk(const boost::filesystem::path& cache_path) {
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

template <concepts::Game Game>
void NeuralNet<Game>::set_model_architecture_signature() {
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

template <concepts::Game Game>
void NeuralNet<Game>::load_data(std::vector<char>& dst, const char* filename) {
  std::ifstream f(filename, std::ios::binary | std::ios::ate);
  size_t sz = f.tellg();
  f.seekg(0);
  dst.resize(sz);
  f.read(dst.data(), sz);
}

template <concepts::Game Game>
void NeuralNet<Game>::load_data(std::vector<char>& dst, std::ispanstream& bytes) {
  dst.assign(std::istreambuf_iterator<char>(bytes), std::istreambuf_iterator<char>());
}

}  // namespace core
