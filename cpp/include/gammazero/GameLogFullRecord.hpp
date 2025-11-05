#pragma once

#include "alphazero/GameLogFullRecord.hpp"
#include "core/concepts/GameConcept.hpp"

namespace gamma0 {

template <core::concepts::Game Game>
struct GameLogFullRecord : public alpha0::GameLogFullRecord<Game> {
  using Base = alpha0::GameLogFullRecord<Game>;
  using ActionValueTensor = Base::ActionValueTensor;
  using WinShareTensor = Game::Types::WinShareTensor;

  ActionValueTensor action_value_uncertainties;  // only valid if action_value_uncertainties_valid
  WinShareTensor Q_posterior;
  WinShareTensor Q_min;  // for each player, the minimum value of Q ever observed for that player
  WinShareTensor Q_max;  // for each player, the maximum value of Q ever observed for that player
  WinShareTensor W_max;  // for each player, the maximum uncertainty ever observed for that player
  bool action_value_uncertainties_valid;
};

}  // namespace gamma0
