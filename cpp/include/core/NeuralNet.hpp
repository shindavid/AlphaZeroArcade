#pragma once

#include <core/BasicTypes.hpp>

#include <boost/filesystem.hpp>
#include <torch/script.h>

#include <sstream>
#include <vector>

namespace core {

/*
 * A thin wrapper around a PyTorch model.
 */
class NeuralNet {
 public:
  using input_vec_t = std::vector<torch::jit::IValue>;

  NeuralNet() : device_(at::Device("cpu")) {}

  /*
   * value is passed to torch::jit::load(). See torch::jit::load() API for details.
   *
   * After this call, the model will be on the CPU. Use activate() to move it to the GPU.
   */
  template<typename Value>
  void load_weights(Value&& value, const std::string& cuda_device);

  // TODO: we need to pass all 2*K+1 torch::Tensor's here! - vector<torch::Tensor>
  void predict(const input_vec_t& input, torch::Tensor& value,
               torch::Tensor& policy, torch::Tensor& action_values) const;

  /*
   * Moves the model to the CPU. This frees up the GPU for other processes.
   */
  void deactivate();

  /*
   * Moves the model back to the GPU.
   */
  void activate();

  bool loaded() const { return loaded_; }
  bool activated() const { return activated_; }

 private:
  mutable torch::jit::script::Module module_;
  at::Device device_;
  bool loaded_ = false;
  bool activated_ = false;
};

}  // namespace core

#include <inline/core/NeuralNet.inl>
