#include <common/NeuralNet.hpp>

namespace common {

inline NeuralNet::NeuralNet(const boost::filesystem::path& path)
    : module_(torch::jit::load(path.c_str()))
{
  module_.to(torch::kCUDA);
}

inline void NeuralNet::predict(const input_vec_t& input, torch::Tensor& policy, torch::Tensor& value) const {
  auto outputs = module_.forward(input).toTuple();
  policy.copy_(outputs->elements()[0].toTensor());
  value.copy_(outputs->elements()[1].toTensor());
}

}  // namespace common
