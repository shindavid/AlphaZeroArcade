#pragma once

#include "core/BasicTypes.hpp"
#include "core/InputTensorizor.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "util/LoggingUtil.hpp"
#include "util/TensorRtUtil.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <unsupported/Eigen/CXX11/Tensor>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <deque>
#include <spanstream>
#include <string>
#include <vector>

namespace core {

struct NeuralNetParams {
  int cuda_device_id;
  int batch_size;
  uint64_t workspace_size_in_bytes;
  trt_util::Precision precision;
};

// Base class for NeuralNet<EvalSpec>
class NeuralNetBase {
 public:
  NeuralNetBase(const NeuralNetParams& params);

  template <typename T>
  void load_weights(T&& onnx_data);

  bool loaded() const { return !plan_data_.empty(); }

 protected:
  // simple logger
  class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
      if (severity <= Severity::kWARNING) {
        LOG_WARN("[TRT] {}", msg);
      }
    }
  };

  void set_model_architecture_signature();
  void load_data(std::vector<char>& dst, const char* filename);
  void load_data(std::vector<char>& dst, std::ispanstream& bytes);

  void init_engine_from_plan_data();
  void refit_engine_plan();
  void build_engine_plan_from_scratch();
  void save_plan_bytes();
  void write_plan_to_disk(const boost::filesystem::path& cache_path);

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;

  std::vector<char> onnx_bytes_;
  std::vector<char> plan_data_;
  std::string model_architecture_signature_;

  const NeuralNetParams params_;
};

/*
 * A thin wrapper around a TensorRT engine.
 */
template <core::concepts::EvalSpec EvalSpec>
class NeuralNet : public NeuralNetBase {
 public:
  using Game = EvalSpec::Game;
  using InputTensorizor = core::InputTensorizor<Game>;
  using TrainingTargets = EvalSpec::TrainingTargets;

  using InputShape = InputTensorizor::Tensor::Dimensions;
  using PolicyShape = Game::Types::PolicyShape;
  using ValueShape = Game::Types::ValueShape;
  using ActionValueShape = Game::Types::ActionValueShape;

  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueTensor = TrainingTargets::ValueTarget::Tensor;
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

  using NeuralNetBase::NeuralNetBase;
  ~NeuralNet();

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

  bool activated() const { return activated_; }

 private:
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

  std::vector<Pipeline*> pipelines_;
  std::deque<pipeline_index_t> available_pipeline_indices_;

  bool activated_ = false;

  mutable mit::mutex pipeline_mutex_;
  mit::condition_variable pipeline_cv_;
};

}  // namespace core

#include "inline/core/NeuralNet.inl"
