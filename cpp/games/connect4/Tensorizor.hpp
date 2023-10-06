#pragma once

#include <torch/torch.h>

#include <core/DerivedTypes.hpp>
#include <core/TensorizorConcept.hpp>
#include <games/connect4/Constants.hpp>
#include <games/connect4/GameState.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/MetaProgramming.hpp>

namespace c4 {

class OwnershipTarget {
 public:
  static constexpr const char* kName = "ownership";
  static constexpr bool kApplySymmetry = true;
  using Shape = eigen_util::Shape<kNumColumns, kNumRows>;
  using Tensor = Eigen::TensorFixedSize<int, Shape, Eigen::RowMajor>;

  static void tensorize(Tensor& tensor, const GameState& state, core::seat_index_t cp) {
    for (int row = 0; row < kNumRows; ++row) {
      for (int col = 0; col < kNumColumns; ++col) {
        core::seat_index_t p = state.get_player_at(row, col);
        int val = (p == -1) ? 0 : ((p == cp) ? 2 : 1);
        tensor(col, row) = val;
      }
    }
  }
};

class Tensorizor {
public:
  using InputShape = eigen_util::Shape<kNumPlayers, kNumColumns, kNumRows>;
  using InputTensor = Eigen::TensorFixedSize<bool, InputShape, Eigen::RowMajor>;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using Action = GameStateTypes::Action;

  using AuxTargetList = mp::TypeList<OwnershipTarget>;

  void clear() {}
  void receive_state_change(const GameState& state, const Action& action) {}

  void tensorize(InputTensor& tensor, const GameState& state) const {
    core::seat_index_t cp = state.get_current_player();
    for (int row = 0; row < kNumRows; ++row) {
      for (int col = 0; col < kNumColumns; ++col) {
        core::seat_index_t p = state.get_player_at(row, col);
        tensor(0, col, row) = (p == cp);
        tensor(1, col, row) = (p == 1 - cp);
      }
    }
  }
};

}  // namespace c4

static_assert(core::TensorizorConcept<c4::Tensorizor, c4::GameState>);
