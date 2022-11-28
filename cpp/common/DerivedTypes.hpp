#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace common {

template<typename GameState>
struct GameStateTypes {
  static constexpr int kNumPlayers = GameState::kNumPlayers;

  using Result = Eigen::TensorFixedSize<float, Eigen::Sizes<kNumPlayers>>;
};

template<typename Tensorizor>
struct TensorizorTypes {
  using TensorShape = util::concat_int_sequence_t<
      util::int_sequence<1>, typename Tensorizor::Shape>;

  using InputTensor = eigen_util::fixed_tensor_t<float, TensorShape>;
};

}  // namespace common
