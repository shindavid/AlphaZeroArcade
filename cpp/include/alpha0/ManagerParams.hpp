#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/ManagerParamsBase.hpp"

namespace alpha0 {

// For now, most of the code lives in ManagerParamsBase, because beta0 is currently just a copy of
// alpha0. As we specialize beta0 more, we should move more code from ManagerParamsBase to
// alpha0::ManagerParams.
template <core::concepts::EvalSpec EvalSpec>
struct ManagerParams : public search::ManagerParamsBase<EvalSpec> {
  using Base = search::ManagerParamsBase<EvalSpec>;

  ManagerParams(search::Mode);

  auto make_options_description();
  bool operator==(const ManagerParams& other) const = default;

  float starting_root_softmax_temperature = 1.4;
  float ending_root_softmax_temperature = 1.1;
  float root_softmax_temperature_half_life = 0.5 * EvalSpec::MctsConfiguration::kOpeningLength;
  float cPUCT = 1.1;
  float cFPU = 0.2;
  float dirichlet_mult = 0.25;

  /*
   * For dirichlet noise, we use a uniform alpha = dirichlet_alpha_factor / sqrt(num_actions).
   */
  float dirichlet_alpha_factor = 0.57;  // ~= .03 * sqrt(361) to match AlphaGo
  bool forced_playouts = true;
  bool enable_first_play_urgency = false;
  float k_forced = 2.0;

  /*
   * These bools control both MCTS dynamics and the zeroing out of the MCTS counts exported to the
   * player (which in turn is exported as a policy training target).
   */
  bool exploit_proven_winners = true;
  bool avoid_proven_losers = true;
};

}  // namespace alpha0

#include "inline/alpha0/ManagerParams.inl"
