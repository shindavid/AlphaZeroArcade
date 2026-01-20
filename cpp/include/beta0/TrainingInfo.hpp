#pragma once

#include "alpha0/TrainingInfo.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct TrainingInfo : public alpha0::TrainingInfo<Game> {
  using Base = alpha0::TrainingInfo<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using WinShareTensor = Game::Types::WinShareTensor;

  void clear() { *this = TrainingInfo(); }

  ActionValueTensor AU;
  WinShareTensor Q;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  WinShareTensor W;
  bool AU_valid = false;
};

}  // namespace beta0
