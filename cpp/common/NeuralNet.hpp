#pragma once

#include <vector>

#include <boost/filesystem.hpp>
#include <torch/script.h>

namespace common {

class NeuralNet {
public:
  using input_vec_t = std::vector<torch::jit::IValue>;

  NeuralNet(const boost::filesystem::path& path);
  void predict(const input_vec_t& input, torch::Tensor& policy, torch::Tensor& value) const;

private:
  mutable torch::jit::script::Module module_;
};

}  // namespace common

#include <common/inl/NeuralNet.inl>
