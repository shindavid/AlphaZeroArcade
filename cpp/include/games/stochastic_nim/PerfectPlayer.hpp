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
  float get_state_value(int stones_left) const { return Qb[stones_left]; }
  int get_optimal_action(int stones_left) const;

 private:
  void iterate();

  using FVector = Eigen::Vector<float, stochastic_nim::kStartingStones + 1>;
  using IVector = Eigen::Vector<int, stochastic_nim::kStartingStones + 1>;
  // V[k]: expected win-rate if there are k stones before a player's move
  FVector Qa;
  // V[k]: expected win-rate if there are k stones after a player's move
  FVector Qb;
  // P[k]: optimal number of stones to take if there are k stones left
  IVector P;
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

