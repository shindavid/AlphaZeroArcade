#include <core/NeuralNet.hpp>

#include <util/Asserts.hpp>
#include <util/CudaUtil.hpp>
#include <util/EigenUtil.hpp>

namespace core {

template <concepts::Game Game>
NeuralNet<Game>::NeuralNet(int batch_size)
    : runtime_(nvinfer1::createInferRuntime(logger_)), batch_size_(batch_size) {}

template <concepts::Game Game>
NeuralNet<Game>::~NeuralNet() {
  deactivate();
  delete runtime_;
}

template <concepts::Game Game>
void NeuralNet<Game>::load_weights(const char* filename, const std::string& cuda_device) {
  plan_data_.clear();

  std::ifstream f(filename, std::ios::binary|std::ios::ate);
  size_t sz = f.tellg(); f.seekg(0);
  plan_data_.resize(sz);
  f.read(plan_data_.data(), sz);

  device_id_ = cuda_util::get_device_id(cuda_device);
  loaded_ = true;
}

template <concepts::Game Game>
void NeuralNet<Game>::load_weights(std::ispanstream& stream, const std::string& cuda_device) {
  plan_data_.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());

  device_id_ = cuda_util::get_device_id(cuda_device);
  loaded_ = true;
}

template <concepts::Game Game>
void NeuralNet<Game>::predict(const DynamicInputTensor& input, DynamicPolicyTensor& policy,
                              DynamicValueTensor& value,
                              DynamicActionValueTensor& action_values) const {
  util::release_assert(activated_, "NeuralNet<Game>::predict() called while deactivated");
  cuda_util::assert_on_device(device_id_);  // TODO: remove this check

  size_t n_input_bytes = input.size() * sizeof(float);
  cudaMemcpy(device_buffers_[0], input.data(), n_input_bytes, cudaMemcpyHostToDevice);

  bool ok = context_->executeV2(device_buffers_.data());
  if (!ok) throw std::runtime_error("TensorRT inference failed");

  copy_output_from_gpu(1, policy);
  copy_output_from_gpu(2, value);
  copy_output_from_gpu(3, action_values);
}

template <concepts::Game Game>
void NeuralNet<Game>::deactivate() {
  if (!activated_) return;

  for (void* ptr : device_buffers_) {
    if (ptr) cudaFree(ptr);
  }
  device_buffers_.clear();

  delete context_;
  delete engine_;
  context_ = nullptr;
  engine_ = nullptr;

  activated_ = false;
}

template <concepts::Game Game>
void NeuralNet<Game>::activate() {
  if (activated_) return;

  util::release_assert(context_ == nullptr, "NeuralNet: illegal {}() call", __func__);

  cudaSetDevice(device_id_);

  engine_ = runtime_->deserializeCudaEngine(plan_data_.data(), plan_data_.size());
  context_ = engine_->createExecutionContext();

  // Since we're using runtime-specified batch size:
  context_->setOptimizationProfileAsync(0, 0);
  nvinfer1::Dims input_shape =
    engine_->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kOPT);

  input_shape.d[0] = batch_size_;
  if (!context_->setInputShape("input", input_shape)) throw std::runtime_error("bad input shape");

  int n_tensors = engine_->getNbIOTensors();
  util::release_assert(n_tensors == 4,
                       "NeuralNet: model must have 4 I/O tensors ({} found)", n_tensors);

  util::release_assert(device_buffers_.empty(),
                       "NeuralNet: device buffers must be empty before {}() call", __func__);

  init_buffer<InputShape>("input", true);
  init_buffer<PolicyShape>("policy");
  init_buffer<ValueShape>("value");
  init_buffer<ActionValueShape>("action_value");

  util::release_assert((int)device_buffers_.size() == n_tensors,
                       "NeuralNet: expected {} device buffers, found {}", n_tensors,
                       device_buffers_.size());

  activated_ = true;
}

template <concepts::Game Game>
template <typename TensorT>
void NeuralNet<Game>::copy_output_from_gpu(int index, TensorT& tensor) const {
  cuda_util::gpu2cpu_memcpy(tensor.data(), device_buffers_[index], tensor.size() * sizeof(float));
}

template <concepts::Game Game>
template <eigen_util::concepts::Shape Shape>
void NeuralNet<Game>::init_buffer(const std::string& expected_name, bool validate_dims) {
  int index = device_buffers_.size();

  const char* name = engine_->getIOTensorName(index);
  util::release_assert(expected_name == name,
                       "NeuralNet: I/O tensor {} must be named '{}', found '{}'", index,
                       expected_name, name);

  auto min_dims = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
  auto max_dims = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);

  constexpr int rank = eigen_util::extract_rank_v<Shape>;

  LOG_INFO(
    std::format("NeuralNet: initializing buffer '{}' at index {} with min_dims={} and max_dims={}",
                name, index, min_dims, max_dims));

  if (validate_dims) {
    util::release_assert(min_dims.nbDims == rank + 1,
                        "Unexpected min dims for '{}': {} ({} != {} + 1)", name, min_dims,
                        min_dims.nbDims, rank);
    util::release_assert(max_dims.nbDims == rank + 1,
                        "Unexpected max dims for '{}': {} ({} != {} + 1)", name, max_dims,
                        max_dims.nbDims, rank);

    Shape shape;
    for (int dim = 0; dim < rank; ++dim) {
      int expected_dim = shape[dim];
      int min_dim = min_dims.d[dim + 1];
      int max_dim = max_dims.d[dim + 1];
      util::release_assert(min_dim == expected_dim,
                          "NeuralNet: min dim {} for '{}' must be {}, found {} ({})", dim + 1, name,
                          expected_dim, min_dim, min_dims);
      util::release_assert(max_dim == expected_dim,
                            "NeuralNet: max dim {} for '{}' must be {}, found {} ({})", dim + 1, name,
                            expected_dim, max_dim, max_dims);
    }
  }

  size_t count = batch_size_ * Shape::TotalSize();
  void* buffer = cuda_util::gpu_malloc(count * sizeof(float));
  device_buffers_.push_back(buffer);

  LOG_INFO("NeuralNet: initialized buffer '{}' at index {}", name, index);
}

}  // namespace core
