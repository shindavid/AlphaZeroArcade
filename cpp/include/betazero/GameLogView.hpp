#pragma once

#include "alphazero/GameLogView.hpp"
#include "core/concepts/GameConcept.hpp"

namespace beta0 {

template <core::concepts::Game Game>
struct GameLogView : public alpha0::GameLogView<Game> {
  using Base = alpha0::GameLogView<Game>;
  using ActionValueTensor = Base::ActionValueTensor;

  ActionValueTensor action_value_uncertainties;
  float Q_prior;
  float Q_posterior;
  bool action_value_uncertainties_valid;
};

}  // namespace beta0
