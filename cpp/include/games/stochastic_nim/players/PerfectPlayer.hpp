#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <games/stochastic_nim/Constants.hpp>
#include <games/stochastic_nim/Game.hpp>
#include <util/EigenUtil.hpp>

namespace stochastic_nim {

class PerfectStrategy {
 public:
  PerfectStrategy();
  float get_state_value_before(int stones_left) const { return Qa[stones_left]; }
  float get_state_value_after(int stones_left) const { return Qb[stones_left]; }
  int get_optimal_action(int stones_left) const;

 private:
  void init_boundary_conditions();
  void iterate();

  using FVector = Eigen::Vector<float, stochastic_nim::kStartingStones + 1>;
  using IVector = Eigen::Vector<int, stochastic_nim::kStartingStones + 1>;
  /*
   * Qa[k]: expected win-rate if there are k stones before a player's move
   * Qb[k]: expected win-rate if there are k stones after a player's move
   * P[k]: optimal number of stones to take if there are k stones left
   */
  FVector Qa;
  FVector Qb;
  IVector P;
};

class PerfectPlayer : public core::AbstractPlayer<stochastic_nim::Game> {
 public:
  using base_t = core::AbstractPlayer<stochastic_nim::Game>;

  struct Params {
    /*
     * The strength parameter controls how well the player plays. It is either 0 (random) or
     * 1 (perfect).
     */
    int strength = 1;
    bool verbose = false;
    auto make_options_description();
  };

  PerfectPlayer(const Params& params, const PerfectStrategy* strategy)
      : params_(params), strategy_(strategy) {}
  ActionResponse get_action_response(const ActionRequest& request) override;

 private:
  const Params params_;
  const PerfectStrategy* strategy_;
};

} // namespace stochastic_nim

#include <inline/games/stochastic_nim/players/PerfectPlayer.inl>
