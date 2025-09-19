#pragma once

#include "alphazero/GameLogFullRecord.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct GameLogFullRecord : public alpha0::GameLogFullRecord<Game> {
  using Base = alpha0::GameLogFullRecord<Game>;
  using ActionValueTensor = Base::ActionValueTensor;

  ActionValueTensor action_value_uncertainties;  // only valid if action_value_uncertainties_valid
  float Q_prior;
  float Q_posterior;
  bool action_value_uncertainties_valid;
};

}  // namespace beta0
