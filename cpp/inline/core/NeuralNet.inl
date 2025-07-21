#include "core/NeuralNet.hpp"

#include "util/Asserts.hpp"
#include "util/CudaUtil.hpp"
#include "util/EigenUtil.hpp"

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
NeuralNet<Game>::NeuralNet(int cuda_device_id)
    : runtime_(nvinfer1::createInferRuntime(logger_)),
      cuda_device_id_(cuda_device_id) {}

template <concepts::Game Game>
NeuralNet<Game>::~NeuralNet() {
  deactivate();

  for (Pipeline* pipeline : pipelines_) {
    delete pipeline;
  }
  delete engine_;
  delete runtime_;
}

template <concepts::Game Game>
void NeuralNet<Game>::load_weights(const char* filename) {
  plan_data_.clear();

  std::ifstream f(filename, std::ios::binary|std::ios::ate);
  size_t sz = f.tellg(); f.seekg(0);
  plan_data_.resize(sz);
  f.read(plan_data_.data(), sz);
}

template <concepts::Game Game>
void NeuralNet<Game>::load_weights(std::ispanstream& stream) {
  plan_data_.clear();
  plan_data_.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());

  if (!activated()) return;
}

template <concepts::Game Game>
pipeline_index_t NeuralNet<Game>::get_pipeline_assignment() {
  mit::unique_lock lock(pipeline_mutex_);
  pipeline_cv_.wait(lock, [&] {
    return !available_pipeline_indices_.empty();
  });
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

  batch_size_ = 0;
}

template <concepts::Game Game>
bool NeuralNet<Game>::activate(int num_pipelines) {
  if (activated()) return false;

  LOG_DEBUG("Activating NeuralNet ({})...", num_pipelines);

  RELEASE_ASSERT(loaded(), "NeuralNet<Game>::{}() called before weights loaded", __func__);

  cuda_util::set_device(cuda_device_id_);
  engine_ = runtime_->deserializeCudaEngine(plan_data_.data(), plan_data_.size());

  nvinfer1::Dims input_shape =
    engine_->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kOPT);
  batch_size_ = input_shape.d[0];

  RELEASE_ASSERT(pipelines_.empty());

  {
    mit::unique_lock lock(pipeline_mutex_);
    RELEASE_ASSERT(available_pipeline_indices_.empty());
    for (int i = 0; i < num_pipelines; ++i) {
      pipelines_.push_back(new Pipeline(engine_, input_shape, batch_size_));
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

}  // namespace core
