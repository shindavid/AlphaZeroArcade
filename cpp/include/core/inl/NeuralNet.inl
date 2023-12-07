#include <core/NeuralNet.hpp>

namespace core {

inline NeuralNet::NeuralNet(const boost::filesystem::path& path, const std::string& cuda_device)
    : module_(torch::jit::load(path.c_str())) {
  module_.to(at::Device(cuda_device));
}

inline void NeuralNet::predict(const input_vec_t& input, torch::Tensor& policy,
                               torch::Tensor& value) const {
  torch::NoGradGuard no_grad;

  auto outputs = module_.forward(input).toTuple();
  policy.copy_(outputs->elements()[0].toTensor());
  value.copy_(outputs->elements()[1].toTensor());
}

}  // namespace core
