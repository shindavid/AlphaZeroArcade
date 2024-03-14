#pragma once

#include <sstream>
#include <vector>

#include <boost/filesystem.hpp>
#include <torch/script.h>

namespace core {

/*
 * A thin wrapper around a PyTorch model.
 */
class NeuralNet {
 public:
  using input_vec_t = std::vector<torch::jit::IValue>;

  /*
   * value is passed to torch::jit::load(). See torch::jit::load() API for details.
   */
  template<typename Value>
  void load_weights(Value&& value, const std::string& cuda_device);

  void predict(const input_vec_t& input, torch::Tensor& policy, torch::Tensor& value) const;

  bool loaded() const { return loaded_; }

 private:
  mutable torch::jit::script::Module module_;
  bool loaded_ = false;
};

}  // namespace core

#include <inline/core/NeuralNet.inl>
