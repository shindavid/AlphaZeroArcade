#pragma once

#include <core/TensorizorConcept.hpp>
#include <games/othello/Constants.hpp>
#include <games/othello/GameState.hpp>
#include <util/EigenUtil.hpp>
#include <util/MetaProgramming.hpp>

namespace othello {

class ScoreMarginPdfTarget {
 public:
  static constexpr const char* kName = "score_margin_pdf";
  static constexpr bool kApplySymmetry = false;
  using Shape = eigen_util::Shape<kNumCells * 2 + 1>;
  using Tensor = Eigen::TensorFixedSize<float, Shape, Eigen::RowMajor>;

  static void tensorize(Tensor& tensor, const GameState& state) {
    int score_margin = state.get_count(kBlack) - state.get_count(kWhite);
    tensor.setZero();
    tensor(score_margin + kNumCells) = 1.0f;
  }

  static void transform(Tensor& tensor, core::symmetry_index_t sym) {}
};

class ScoreMarginCdfTarget {
 public:
  static constexpr const char* kName = "score_margin_cdf";
  static constexpr bool kApplySymmetry = false;
  using Shape = eigen_util::Shape<kNumCells * 2 + 1>;
  using Tensor = Eigen::TensorFixedSize<float, Shape, Eigen::RowMajor>;

  static void tensorize(Tensor& tensor, const GameState& state) {
    int score_margin = state.get_count(kBlack) - state.get_count(kWhite);
    tensor.setZero();
    for (int i = 0; i <= score_margin + kNumCells; ++i) {
      tensor(i) = 1.0f;
    }
  }

  static void transform(Tensor& tensor, core::symmetry_index_t sym) {}
};

class OwnershipTarget {
 public:
  static constexpr const char* kName = "ownership";
  static constexpr bool kApplySymmetry = true;
  using Shape = eigen_util::Shape<kBoardDimension, kBoardDimension>;
  using Tensor = Eigen::TensorFixedSize<float, Shape, Eigen::RowMajor>;

  static void tensorize(Tensor& tensor, const GameState& state) {
    for (int row = 0; row < kBoardDimension; ++row) {
      for (int col = 0; col < kBoardDimension; ++col) {
        tensor(row, col) = state.get_player_at(row, col);
      }
    }
  }
};

class Tensorizor {
 public:
  using InputShape = eigen_util::Shape<kNumPlayers, kBoardDimension, kBoardDimension>;
  using InputTensor = Eigen::TensorFixedSize<bool, InputShape, Eigen::RowMajor>;

  using GameStateTypes = core::GameStateTypes<GameState>;
  using Action = GameStateTypes::Action;

  void clear() {}
  void receive_state_change(const GameState& state, const Action& action) {}

  using AuxTargetList = mp::TypeList <
    ScoreMarginPdfTarget,
    ScoreMarginCdfTarget,
    OwnershipTarget
  >;

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
