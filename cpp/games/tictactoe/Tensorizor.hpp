#pragma once

#include <core/TensorizorConcept.hpp>
#include <games/tictactoe/Constants.hpp>
#include <games/tictactoe/GameState.hpp>
#include <util/EigenUtil.hpp>
#include <util/MetaProgramming.hpp>

namespace tictactoe {

class OwnershipTarget {
 public:
  static constexpr const char* kName = "ownership";
  static constexpr bool kApplySymmetry = true;
  using Shape = eigen_util::Shape<kBoardDimension, kBoardDimension>;
  using Tensor = Eigen::TensorFixedSize<int, Shape, Eigen::RowMajor>;

  static void tensorize(Tensor& tensor, const GameState& state) {
    for (int row = 0; row < kBoardDimension; ++row) {
      for (int col = 0; col < kBoardDimension; ++col) {
        tensor(row, col) = 1 + state.get_player_at(row, col);
      }
    }
  }
};

class Tensorizor {
 public:
  using InputShape = eigen_util::Shape<kNumPlayers, kBoardDimension, kBoardDimension>;
  using InputTensor = Eigen::TensorFixedSize<bool, InputShape, Eigen::RowMajor>;

  using AuxTargetList = mp::TypeList<OwnershipTarget>;

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
