#pragma once

#include "alphazero/TrainingInfo.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct TrainingInfo : public alpha0::TrainingInfo<Game> {
  using Base = alpha0::TrainingInfo<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using ValueTensor = Game::Types::ValueTensor;

  void clear() { *this = TrainingInfo(); }

  ActionValueTensor action_value_uncertainties_target;
  ValueTensor Q_posterior;
  bool action_value_uncertainties_target_valid = false;
};

}  // namespace beta0
