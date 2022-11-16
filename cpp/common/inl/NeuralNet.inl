#include <common/NeuralNet.hpp>

namespace common {

inline NeuralNet::NeuralNet(const boost::filesystem::path& path)
    : module_(torch::jit::load(path.c_str())) {}

inline void NeuralNet::predict(const input_vec_t& input, torch::Tensor& policy, torch::Tensor& value) {
  auto outputs = module_.forward(input).toTuple();
  policy = outputs->elements()[0].toTensor();
  value = outputs->elements()[1].toTensor();
}

}  // namespace common
