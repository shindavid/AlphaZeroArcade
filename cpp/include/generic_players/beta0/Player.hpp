#pragma once

#include "core/ActionResponse.hpp"
#include "generic_players/alpha0/Player.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace generic::beta0 {

/*
 * The generic::beta0::Player uses BetaZero MCTS to select actions.
 *
 * Note that when 2 or more identically-configured generic::beta0::Player's are playing in the same
 * game, they can share the same MCTS tree, as an optimization. This implementation supports this
 * optimization.
 */
template <search::concepts::Traits Traits_>
class Player : public generic::alpha0::Player<Traits_> {
 public:
  using Traits = Traits_;
  using Game = Traits::Game;
  using EvalSpec = Traits::EvalSpec;
  using base_t = generic::alpha0::Player<Traits>;
  using ActionRequest = base_t::ActionRequest;
  using SearchResults = base_t::SearchResults;

  using ActionMask = Game::Types::ActionMask;
  using PolicyTensor = Game::Types::PolicyTensor;

  using base_t::base_t;

 protected:
  virtual core::ActionResponse get_action_response_helper(const SearchResults*,
                                                          const ActionRequest&) override;

  auto get_action_policy(const SearchResults*, const ActionMask&) const;
};

}  // namespace generic::beta0

#include "inline/generic_players/beta0/Player.inl"
