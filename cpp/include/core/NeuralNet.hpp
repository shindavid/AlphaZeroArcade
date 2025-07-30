#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/Game.hpp"
#include "util/LoggingUtil.hpp"
#include "util/mit/mit.hpp"

#include <Eigen/Core>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <deque>
#include <spanstream>
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

  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueTensor = Game::TrainingTargets::ValueTarget::Tensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  using DynamicInputTensor = Eigen::Tensor<float, InputShape::count + 1, Eigen::RowMajor>;
  using DynamicPolicyTensor = Eigen::Tensor<float, PolicyShape::count + 1, Eigen::RowMajor>;
  using DynamicValueTensor = Eigen::Tensor<float, ValueShape::count + 1, Eigen::RowMajor>;
  using DynamicActionValueTensor =
    Eigen::Tensor<float, ActionValueShape::count + 1, Eigen::RowMajor>;

  using DynamicInputTensorMap = Eigen::TensorMap<DynamicInputTensor, Eigen::Aligned>;
  using DynamicPolicyTensorMap = Eigen::TensorMap<DynamicPolicyTensor, Eigen::Aligned>;
  using DynamicValueTensorMap = Eigen::TensorMap<DynamicValueTensor, Eigen::Aligned>;
  using DynamicActionValueTensorMap = Eigen::TensorMap<DynamicActionValueTensor, Eigen::Aligned>;

  NeuralNet(int cuda_device_id);
  ~NeuralNet();

  int batch_size() const { return batch_size_; }
  void load_weights(const char* filename);
  void load_weights(std::ispanstream& stream);

  pipeline_index_t get_pipeline_assignment();
  float* get_input_ptr(pipeline_index_t);
  void schedule(pipeline_index_t) const;
  void release(pipeline_index_t);

  void load(pipeline_index_t, float** policy_data, float** value_data, float** action_values_data);

  // Frees all GPU resources
  void deactivate();

  // If already activated, is a no-op and returns false.
  //
  // Else, sets up GPU resources, including pipelines, and returns true.
  //
  // Must be called by the thread doing the {get_pipeline_assignment(), schedule()} calls.
  bool activate(int num_pipelines);

  bool loaded() const { return !plan_data_.empty(); }
  bool activated() const { return engine_; }

 private:
  // simple logger
  class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
      if (severity <= Severity::kWARNING) {
        LOG_WARN("[TRT] {}", msg);
      }
    }
  };

  struct Pipeline {
    Pipeline(nvinfer1::ICudaEngine* engine, const nvinfer1::Dims& input_shape, int batch_size);
    ~Pipeline();

    void schedule();
    void load(float** policy_data, float** value_data, float** action_values_data);

    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream;
    std::vector<void*> device_buffers;

    DynamicInputTensorMap input;
    DynamicPolicyTensorMap policy;
    DynamicValueTensorMap value;
    DynamicActionValueTensorMap action_values;
  };

  Logger logger_;
  nvinfer1::IRuntime* const runtime_;
  nvinfer1::ICudaEngine* engine_ = nullptr;

  std::vector<Pipeline*> pipelines_;
  std::deque<pipeline_index_t> available_pipeline_indices_;
  std::vector<char> plan_data_;

  mutable mit::mutex pipeline_mutex_;
  mit::condition_variable pipeline_cv_;

  int batch_size_ = 0;
  const int cuda_device_id_;
};

}  // namespace core

#include "inline/core/NeuralNet.inl"
