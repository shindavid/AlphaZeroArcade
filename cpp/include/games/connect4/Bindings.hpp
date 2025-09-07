#pragma once

#include "core/BayesianMctsEvalSpec.hpp"
#include "core/DefaultKeys.hpp"
#include "core/InputTensorizor.hpp"
#include "core/MctsEvalSpec.hpp"
#include "core/TrainingTargets.hpp"
#include "games/connect4/Game.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/MetaProgramming.hpp"

namespace c4 {

struct InputTensorizor {
  static constexpr int kDim0 = kNumPlayers * (1 + Game::Constants::kNumPreviousStatesToEncode);
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kNumRows, kNumColumns>>;

  template <util::concepts::RandomAccessIteratorOf<Game::State> Iter>
  static Tensor tensorize(Iter start, Iter cur) {
    core::seat_index_t cp = Game::Rules::get_current_player(*cur);
    Tensor tensor;
    tensor.setZero();
    int i = 0;
    Iter state = cur;
    while (true) {
      for (int row = 0; row < kNumRows; ++row) {
        for (int col = 0; col < kNumColumns; ++col) {
          core::seat_index_t p = state->get_player_at(row, col);
          if (p < 0) continue;
          int x = (kNumPlayers + cp - p) % kNumPlayers;
          tensor(i + x, row, col) = 1;
        }
      }
      if (state == start) break;
      state--;
      i += kNumPlayers;
    }
    return tensor;
  }
};

}  // namespace c4

namespace core {

template <>
struct InputTensorizor<c4::Game> : public c4::InputTensorizor {
  using Keys = core::DefaultKeys<c4::Game>;
};

}  // namespace core

namespace c4::alpha0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
};

}  // namespace c4::alpha0

namespace core::alpha0 {

template <>
struct EvalSpec<c4::Game> {
  using Game = c4::Game;
  using TrainingTargets = c4::alpha0::TrainingTargets;
};

}  // namespace core::alpha0

namespace c4::beta0 {

struct TrainingTargets {
  using PolicyTarget = core::PolicyTarget<Game>;
  using ValueTarget = core::ValueTarget<Game>;
  using ActionValueTarget = core::ActionValueTarget<Game>;
  using OppPolicyTarget = core::OppPolicyTarget<Game>;

  using List = mp::TypeList<PolicyTarget, ValueTarget, ActionValueTarget, OppPolicyTarget>;
};

}  // namespace c4::beta0

namespace core::beta0 {

template <>
struct EvalSpec<c4::Game> {
  using Game = c4::Game;
  using TrainingTargets = c4::beta0::TrainingTargets;
};

}  // namespace core::beta0
