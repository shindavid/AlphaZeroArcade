#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <games/tictactoe/Constants.hpp>
#include <games/tictactoe/GameState.hpp>
#include <util/BoostUtil.hpp>

namespace tictactoe {

class PerfectPlayer : public Player {
public:
  using base_t = Player;

  struct Params {
    /*
     * The strength parameter controls how well the player plays. It is either 0 (random) or
     * 1 (perfect).
     */
    int strength = 1;
    auto make_options_description();
  };

  PerfectPlayer(const Params&);

  core::action_t get_action(const GameState&, const ActionMask&) override;

private:
  const Params params_;
};

}  // namespace tictactoe

#include <games/tictactoe/players/inl/PerfectPlayer.inl>
