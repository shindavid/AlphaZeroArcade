#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/stochastic_nim/Constants.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <util/BitSet.hpp>
#include <util/EigenUtil.hpp>

namespace stochastic_nim {

class PerfectStrategy {
 public:
  struct Params {
    const int starting_stones;
    const int max_stones_to_take;
    const float* chance_event_probs;
    const int num_chance_events;
  };

  PerfectStrategy(Params params);
  float* get_state_value() const { return state_value_; }
  int* get_optimal_action() const { return optimal_action_; }
  Params get_params() const { return params_; }

 private:
  void iterate();
  Params params_;
  // expected win rate if there are [index up to starting_stones] stones left after a player's move
  float* state_value_;
  // best number of stones to take if there are [index up to starting stones] stones left
  int* optimal_action_;
};

class PerfectPlayer : public core::AbstractPlayer<stochastic_nim::Game> {
 public:

  using base_t = core::AbstractPlayer<stochastic_nim::Game>;

  PerfectPlayer(PerfectStrategy* strategy) : strategy_(strategy) {}
  ActionResponse get_action_response(const State&, const ActionMask&) override;

 private:
  const PerfectStrategy* strategy_;
};

} // namespace stochastic_nim

#include <inline/games/stochastic_nim/PerfectPlayer.inl>

