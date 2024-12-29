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
  PerfectStrategy();
  float get_state_value(int stones_left) const { return state_values_[stones_left]; }
  int get_optimal_action(int stones_left) const { return optimal_actions_[stones_left] - 1; }

 private:
  void iterate();
  using FArray = eigen_util::FArray<stochastic_nim::kStartingStones + 1>;
  // expected win rate if there are [index up to starting_stones] stones left after a player's move
  FArray state_values_;
  // best number of stones to take if there are [index up to starting stones] stones left
  FArray optimal_actions_;
};

class PerfectPlayer : public core::AbstractPlayer<stochastic_nim::Game> {
 public:
  using base_t = core::AbstractPlayer<stochastic_nim::Game>;

  PerfectPlayer(const PerfectStrategy* strategy) : strategy_(strategy) {}
  ActionResponse get_action_response(const State&, const ActionMask&) override;

 private:
  const PerfectStrategy* strategy_;
};

} // namespace stochastic_nim

#include <inline/games/stochastic_nim/PerfectPlayer.inl>

