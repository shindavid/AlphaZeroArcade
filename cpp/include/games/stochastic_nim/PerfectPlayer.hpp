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
  int get_optimal_action(int stones_left) const;

 private:
  void iterate();

  using FArray = eigen_util::FArray<stochastic_nim::kStartingStones + 1>;
  using IArray = Eigen::Array<int, stochastic_nim::kStartingStones + 1, 1>;
  // state_values_[k]: expected win-rate if there are k stones left after a player's move
  FArray state_values_;
  // optimal_actions_[k]: optimal number of stones to take if there are k stones left
  IArray optimal_actions_;
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

