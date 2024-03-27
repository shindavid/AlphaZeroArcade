#include <core/NeuralNet.hpp>

#include <util/Asserts.hpp>

namespace core {

template <typename Value>
inline void NeuralNet::load_weights(Value&& value, const std::string& cuda_device) {
  util::release_assert(!activated_, "NeuralNet::load_weights() called while activated");
  new (&module_) torch::jit::script::Module(torch::jit::load(value));
  device_ = at::Device(cuda_device);
  loaded_ = true;
}

inline void NeuralNet::predict(const input_vec_t& input, torch::Tensor& policy,
                               torch::Tensor& value) const {
  util::release_assert(activated_, "NeuralNet::predict() called while deactivated");
  torch::NoGradGuard no_grad;

  auto outputs = module_.forward(input).toTuple();
  policy.copy_(outputs->elements()[0].toTensor());
  value.copy_(outputs->elements()[1].toTensor());
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
