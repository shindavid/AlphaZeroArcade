#pragma once

#include "core/InputTensorizor.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "util/MetaProgramming.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <array>
#include <tuple>

namespace core {

namespace detail {

template <typename Target>
struct ExtractTensor {
  using type = Target::Tensor;
};

template <typename Tensor>
struct ExtractDimensions {
  using type = Tensor::Dimensions;
};

template <typename Shape>
struct ToDynamicTensor {
  using type = Eigen::Tensor<float, Shape::count + 1, Eigen::RowMajor>;
};

template <typename Tensor>
struct ToTensorMap {
  using type = Eigen::TensorMap<Tensor, Eigen::Aligned>;
};

}  // namespace detail

template <core::concepts::EvalSpec EvalSpec>
struct TensorTypes {
  using Game = EvalSpec::Game;
  using InputTensorizor = core::InputTensorizor<Game>;
  using PrimaryTargets = EvalSpec::TrainingTargets::PrimaryList;

  using InputTensor = InputTensorizor::Tensor;
  using OutputTensors = mp::Apply_t<PrimaryTargets, detail::ExtractTensor>;

  using InputShape = InputTensor::Dimensions;
  using OutputShapes = mp::Apply_t<OutputTensors, detail::ExtractDimensions>;

  using OutputTensorMaps = mp::Apply_t<OutputTensors, detail::ToTensorMap>;

  using DynamicInputTensor = Eigen::Tensor<float, InputShape::count + 1, Eigen::RowMajor>;
  using DynamicOutputTensors = mp::Apply_t<OutputShapes, detail::ToDynamicTensor>;

  using DynamicInputTensorMap = Eigen::TensorMap<DynamicInputTensor, Eigen::Aligned>;
  using DynamicOutputTensorMaps = mp::Apply_t<DynamicOutputTensors, detail::ToTensorMap>;

  using OutputTensorTuple = mp::Rebind_t<OutputTensors, std::tuple>;
  using DynamicOutputTensorMapTuple = mp::Rebind_t<DynamicOutputTensorMaps, std::tuple>;

  static constexpr int kNumOutputs = mp::Length_v<OutputTensors>;
  using OutputDataArray = std::array<float*, kNumOutputs>;
};

}  // namespace core
