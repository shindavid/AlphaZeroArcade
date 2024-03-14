#include <core/NeuralNet.hpp>

namespace core {

template <typename Value>
inline void NeuralNet::load_weights(Value&& value, const std::string& cuda_device) {
  new (&module_) torch::jit::script::Module(torch::jit::load(value));
  module_.to(at::Device(cuda_device));
  loaded_ = true;
}

inline void NeuralNet::predict(const input_vec_t& input, torch::Tensor& policy,
                               torch::Tensor& value) const {
  torch::NoGradGuard no_grad;

  auto outputs = module_.forward(input).toTuple();
  policy.copy_(outputs->elements()[0].toTensor());
  value.copy_(outputs->elements()[1].toTensor());
}

}  // namespace core
