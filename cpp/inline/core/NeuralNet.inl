#include "core/NeuralNet.hpp"

#include "util/Asserts.hpp"
#include "util/CudaUtil.hpp"
#include "util/EigenUtil.hpp"

#include <onnx/onnx_pb.h>

#include <NvInfer.h>

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

// NeuralNetBase

template <typename T>
void NeuralNetBase::load_weights(T&& onnx_data) {
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
      init_engine_from_plan_data();
      refit = true;
    }
  }

  if (refit) {
    if (!engine_) {
      init_engine_from_plan_data();
    }
    refit_engine_plan();
    save_plan_bytes();
  } else {
    build_engine_plan_from_scratch();
    save_plan_bytes();
    write_plan_to_disk(cache_path);
  }
}

// NeuralNet<EvalSpec>

template <core::concepts::EvalSpec EvalSpec>
NeuralNet<EvalSpec>::~NeuralNet() {
  deactivate();

  for (Pipeline* pipeline : pipelines_) {
    delete pipeline;
  }
}

template <core::concepts::EvalSpec EvalSpec>
pipeline_index_t NeuralNet<EvalSpec>::get_pipeline_assignment() {
  mit::unique_lock lock(pipeline_mutex_);
  pipeline_cv_.wait(lock, [&] { return !available_pipeline_indices_.empty(); });
  pipeline_index_t index = available_pipeline_indices_.back();
  available_pipeline_indices_.pop_back();
  return index;
}

template <core::concepts::EvalSpec EvalSpec>
float* NeuralNet<EvalSpec>::get_input_ptr(pipeline_index_t index) {
  return pipelines_[index]->input.data();
}

template <core::concepts::EvalSpec EvalSpec>
void NeuralNet<EvalSpec>::schedule(pipeline_index_t index) const {
  RELEASE_ASSERT(activated(), "NeuralNet<EvalSpec>::predict() called while deactivated");
  pipelines_[index]->schedule();
}

template <core::concepts::EvalSpec EvalSpec>
void NeuralNet<EvalSpec>::release(pipeline_index_t index) {
  mit::unique_lock lock(pipeline_mutex_);
  available_pipeline_indices_.push_back(index);
  lock.unlock();
  pipeline_cv_.notify_all();
}

template <core::concepts::EvalSpec EvalSpec>
void NeuralNet<EvalSpec>::load(pipeline_index_t index, float** policy_data, float** value_data,
                               float** action_values_data) {
  pipelines_[index]->load(policy_data, value_data, action_values_data);
}

template <core::concepts::EvalSpec EvalSpec>
void NeuralNet<EvalSpec>::deactivate() {
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

  engine_.reset();
}

template <core::concepts::EvalSpec EvalSpec>
bool NeuralNet<EvalSpec>::activate(int num_pipelines) {
  if (activated()) return false;

  LOG_DEBUG("Activating NeuralNet ({})...", num_pipelines);

  activated_ = true;
  RELEASE_ASSERT(loaded(), "NeuralNet<EvalSpec>::{}() called before weights loaded", __func__);

  cuda_util::set_device(params_.cuda_device_id);
  if (!engine_) {
    init_engine_from_plan_data();
  }

  nvinfer1::Dims input_shape =
    engine_->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kOPT);

  RELEASE_ASSERT(pipelines_.empty());

  {
    mit::unique_lock lock(pipeline_mutex_);
    RELEASE_ASSERT(available_pipeline_indices_.empty());
    for (int i = 0; i < num_pipelines; ++i) {
      pipelines_.push_back(new Pipeline(engine_.get(), input_shape, params_.batch_size));
      available_pipeline_indices_.push_back(i);
    }
  }
  pipeline_cv_.notify_all();

  LOG_DEBUG("Done activating NeuralNet ({})!", num_pipelines);
  return true;
}

template <core::concepts::EvalSpec EvalSpec>
NeuralNet<EvalSpec>::Pipeline::Pipeline(nvinfer1::ICudaEngine* engine,
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

template <core::concepts::EvalSpec EvalSpec>
NeuralNet<EvalSpec>::Pipeline::~Pipeline() {
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

template <core::concepts::EvalSpec EvalSpec>
void NeuralNet<EvalSpec>::Pipeline::schedule() {
  constexpr size_t f = sizeof(float);
  auto& dbs = device_buffers;
  cuda_util::cpu2gpu_memcpy_async(stream, dbs[0], input.data(), input.size() * f);

  bool ok = context->enqueueV3(stream);
  if (!ok) throw std::runtime_error("TensorRT inference failed");

  cuda_util::gpu2cpu_memcpy_async(stream, policy.data(), dbs[1], policy.size() * f);
  cuda_util::gpu2cpu_memcpy_async(stream, value.data(), dbs[2], value.size() * f);
  cuda_util::gpu2cpu_memcpy_async(stream, action_values.data(), dbs[3], action_values.size() * f);
}

template <core::concepts::EvalSpec EvalSpec>
void NeuralNet<EvalSpec>::Pipeline::load(float** policy_data, float** value_data,
                                         float** action_values_data) {
  cuda_util::synchronize_stream(stream);
  *policy_data = policy.data();
  *value_data = value.data();
  *action_values_data = action_values.data();
}

}  // namespace core
