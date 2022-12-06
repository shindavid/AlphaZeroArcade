#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <util/CppUtil.hpp>
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

  using PolicyVector = Eigen::Vector<float, kNumGlobalActions>;
  using ValueVector = Eigen::Vector<float, kNumPlayers>;
};

template<typename Tensorizor>
struct TensorizorTypes {
  using TensorShape = util::concat_int_sequence_t<util::int_sequence<1>, typename Tensorizor::Shape>;

  using InputTensor = eigen_util::fixed_tensor_t<float, eigen_util::to_sizes_t<TensorShape>>;
};

}  // namespace common
