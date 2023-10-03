#pragma once

#include <core/TensorizorConcept.hpp>
#include <games/othello/Constants.hpp>
#include <games/othello/GameState.hpp>
#include <util/EigenUtil.hpp>

namespace othello {

/*
 * All transforms have a templated transform_input() method. This generality
 * exists to support unit tests, which use non-bool input tensors.
 */
class Tensorizor {
 public:
  using InputShape = eigen_util::Shape<kNumPlayers, kBoardDimension, kBoardDimension>;
  using InputTensor = Eigen::TensorFixedSize<bool, InputShape, Eigen::RowMajor>;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using Action = GameStateTypes::Action;

  void clear() {}
  void receive_state_change(const GameState& state, const Action& action) {}

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

}  // namespace othello

static_assert(core::TensorizorConcept<othello::Tensorizor, othello::GameState>);
