#pragma once

#include "generic_players/x0/Player.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace generic::beta0 {

/*
 * The generic::beta0::Player uses Bayesian Minimax Tree Search to select actions.
 */
template <search::concepts::Traits Traits_>
class Player : public generic::x0::Player<Traits_> {
 public:
  using Base = generic::x0::Player<Traits_>;
  using BasePlayer = Player;  // a little ugly, but needed for generic::x0::PlayerGeneratorBase
  using Traits = Traits_;
  using Game = Traits::Game;
  using BaseParams = Base::Params;

  using Params = BaseParams;
  using Base::Base;

  using SearchResults = Traits::SearchResults;
  using ActionMask = Game::Types::ActionMask;
  using PolicyTensor = Game::Types::PolicyTensor;

 protected:
  virtual PolicyTensor get_action_policy(const SearchResults*, const ActionMask&) const override;
};

}  // namespace generic::beta0

#include "inline/generic_players/beta0/Player.inl"
