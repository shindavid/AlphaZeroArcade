#pragma once

#include "alphazero/TrainingInfo.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct TrainingInfo : public alpha0::TrainingInfo<Game> {
  using Base = alpha0::TrainingInfo<Game>;
  using ActionValueTensor = Base::ActionValueTensor;

  void clear() { *this = TrainingInfo(); }

  ActionValueTensor action_value_uncertainties_target;
  float Q_prior = 0;
  float Q_posterior = 0;
  bool action_value_uncertainties_target_valid = false;
};

}  // namespace beta0
