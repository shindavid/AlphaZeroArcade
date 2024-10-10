#include <core/NeuralNet.hpp>

#include <util/Asserts.hpp>

namespace core {

template <typename Value>
inline void NeuralNet::load_weights(Value&& value, const std::string& cuda_device) {
  util::release_assert(!activated_, "NeuralNet::load_weights() called while activated");

  // TODO: this torch::jit::load() can take 100's of milliseconds. This is perhaps understandable
  // on the first load, but on subsequent loads during self-play, it feels like it should be
  // possible for this to be much faster, since the architecture doesn't change, only the weights.
  //
  // Not a high priority, but if we want to squeeze out an extra 1-2% of GPU utilization, this is
  // one place to look. We are potentially looking to migrate from libtorch to a more mature
  // library (onnx?) - we may get this optimization for free if/when we do that.
  new (&module_) torch::jit::script::Module(torch::jit::load(value));
  device_ = at::Device(cuda_device);
  loaded_ = true;
}

inline void NeuralNet::predict(const input_vec_t& input, torch::Tensor& policy,
                               torch::Tensor& value, torch::Tensor& action_values) const {
  util::release_assert(activated_, "NeuralNet::predict() called while deactivated");
  torch::NoGradGuard no_grad;

  auto outputs = module_.forward(input).toTuple();
  policy.copy_(outputs->elements()[0].toTensor());
  value.copy_(outputs->elements()[1].toTensor());
  action_values.copy_(outputs->elements()[2].toTensor());
}

inline void NeuralNet::deactivate() {
  if (activated_) {
    module_.to(at::Device("cpu"));
    activated_ = false;
  }
}

inline void NeuralNet::activate() {
  if (!activated_) {
    module_.to(device_);
    activated_ = true;
  }
}

}  // namespace core
