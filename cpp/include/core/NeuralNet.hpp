#pragma once

#include <core/concepts/Game.hpp>
#include <util/EigenUtil.hpp>
#include <util/LoggingUtil.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <spanstream>
#include <string>
#include <vector>

namespace core {

/*
 * A thin wrapper around a TensorRT engine.
 */
template <concepts::Game Game>
class NeuralNet {
 public:
  using InputShape = Game::InputTensorizor::Tensor::Dimensions;
  using PolicyShape = Game::Types::PolicyShape;
  using ValueShape = Game::Types::ValueShape;
  using ActionValueShape = Game::Types::ActionValueShape;

  using DynamicInputTensor = Eigen::Tensor<float, InputShape::count + 1, Eigen::RowMajor>;
  using DynamicPolicyTensor = Eigen::Tensor<float, PolicyShape::count + 1, Eigen::RowMajor>;
  using DynamicValueTensor = Eigen::Tensor<float, ValueShape::count + 1, Eigen::RowMajor>;
  using DynamicActionValueTensor =
    Eigen::Tensor<float, ActionValueShape::count + 1, Eigen::RowMajor>;

  NeuralNet(int batch_size);
  ~NeuralNet();

  void load_weights(const char* filename, const std::string& cuda_device);
  void load_weights(std::ispanstream& stream, const std::string& cuda_device);

  void predict(const DynamicInputTensor& input, DynamicPolicyTensor&, DynamicValueTensor&,
               DynamicActionValueTensor&) const;

  // Moves the model to the CPU. This frees up the GPU for other processes.
  void deactivate();

  // Moves the model back to the GPU.
  void activate();

  bool loaded() const { return loaded_; }
  bool activated() const { return activated_; }

 private:
  template <typename TensorT>
  void copy_output_from_gpu(int index, TensorT& tensor) const;

  template <eigen_util::concepts::Shape Shape>
  void init_buffer(const std::string& expected_name, bool validate_dims=false);

  // simple logger
  class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
      if (severity <= Severity::kWARNING) {
        LOG_WARN("[TRT] {}", msg);
      }
    }
  };

  Logger logger_;
  nvinfer1::IRuntime* const runtime_;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  std::vector<char> plan_data_;
  std::vector<void*> device_buffers_;

  const int batch_size_;
  int device_id_ = -1;
  bool loaded_ = false;
  bool activated_ = false;
};

}  // namespace core

#include <inline/core/NeuralNet.inl>
