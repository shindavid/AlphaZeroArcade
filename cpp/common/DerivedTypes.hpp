#pragma once

#include <utility>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <common/BasicTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenTorch.hpp>
#include <util/EigenUtil.hpp>

namespace common {

/*
 * Represents the result of a game, as a length-t array of non-negative floats, where t is the number of players in
 * the game.
 *
 * If the result represents a terminal game state, the array will have sum 1. Normally, one slot in the array,
 * corresponding to the winner, will equal 1, and the other slots will equal 0. In the even of a draw, the tied
 * players will typically each equal the same fractional value.
 *
 * If the game is not yet over, the result will have all zeros.
 */
template<int NumPlayers> using GameResult = Eigen::Vector<float, NumPlayers>;
template<int NumPlayers> bool is_terminal_result(const GameResult<NumPlayers>& result) { return result.sum() > 0; }
template<int NumPlayers> auto make_non_terminal_result() { GameResult<NumPlayers> r; r.setZero(); return r; }

template<typename GameState>
struct GameStateTypes {
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameState::kNumGlobalActions;

  using Result = GameResult<kNumPlayers>;

  template <int NumRows> using PolicyMatrix = eigentorch::Matrix<float, NumRows, kNumGlobalActions, Eigen::RowMajor>;
  template <int NumRows> using ValueMatrix = eigentorch::Matrix<float, NumRows, kNumPlayers, Eigen::RowMajor>;

  using PolicyVector = PolicyMatrix<1>;
  using ValueVector = ValueMatrix<1>;
};

template<typename Tensorizor>
struct TensorizorTypes {
  using BaseShape = typename Tensorizor::Shape;
  using Shape = eigen_util::to_sizes_t<util::concat_int_sequence_t<util::int_sequence<1>, BaseShape>>;
  using InputTensor = eigentorch::TensorFixedSize<float, Shape>;
  using DynamicInputTensor = eigentorch::Tensor<float, BaseShape::size() + 1>;
};

template<typename GameState>
struct StateSymmetryIndex {
  GameState state;
  symmetry_index_t sym_index;

  bool operator==(const StateSymmetryIndex& other) const {
    return state == other.state && sym_index == other.sym_index;
  }
};

}  // namespace common

template <typename GameState>
struct std::hash<common::StateSymmetryIndex<GameState>> {
  std::size_t operator()(const common::StateSymmetryIndex<GameState> ssi) const {
    constexpr size_t some_prime = 1113859;
    return some_prime * std::hash(ssi.state) + ssi.sym_index;
  }
};
