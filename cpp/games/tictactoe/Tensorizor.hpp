#pragma once

#include <core/TensorizorConcept.hpp>
#include <games/tictactoe/Constants.hpp>
#include <games/tictactoe/GameState.hpp>
#include <util/EigenUtil.hpp>

namespace tictactoe {

/*
 * All transforms have a templated transform_input() method. This generality exists to support unit tests, which
 * use non-bool input tensors.
 */
class Tensorizor {
 public:
  using InputShape = eigen_util::Shape<kNumPlayers, kBoardDimension, kBoardDimension>;
  using InputTensor = Eigen::TensorFixedSize<bool, InputShape, Eigen::RowMajor>;

  void clear() {}
  void receive_state_change(const GameState&, const GameState::Action&) {}

  void tensorize(InputTensor& tensor, const GameState& state) const {
    core::seat_index_t cp = state.get_current_player();
    for (int row = 0; row < kBoardDimension; ++row) {
      for (int col = 0; col < kBoardDimension; ++col) {
        core::seat_index_t p = state.get_player_at(row, col);
        tensor(0, row, col) = (p == cp);
        tensor(1, row, col) = (p == 1 - cp);
      }
    }
  }
};

}  // namespace tictactoe

static_assert(core::TensorizorConcept<tictactoe::Tensorizor, tictactoe::GameState>);
