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
  using SharedData_sptr = Base::SharedData_sptr;
  using SearchResults = Traits::SearchResults;
  using ActionMask = Game::Types::ActionMask;
  using PolicyTensor = Game::Types::PolicyTensor;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using ActionRequest = core::ActionRequest<Game>;
  using AuxData = Traits::AuxData;
  using VerboseData = Traits::VerboseData;
  using State = Game::State;
  using GameResultTensor = Game::GameResults::Tensor;

  struct ParamsExtra {
    bool verbose = false;
  };

  struct Params : public BaseParams, public ParamsExtra {
    using BaseParams::BaseParams;

    auto make_options_description();
  };

  void end_game(const State& state, const GameResultTensor& results) override;

  Player(const Params& params, SharedData_sptr shared_data, bool owns_shared_data)
      : Base(params, shared_data, owns_shared_data), params_extra_(params) {}

 protected:
  virtual PolicyTensor get_action_policy(const SearchResults*, const ActionMask&) const override;
  virtual core::ActionResponse get_action_response_helper(const SearchResults*,
                                                          const ActionRequest&) override;
 private:
  ParamsExtra params_extra_;
  std::vector<AuxData*> aux_data_ptrs_;
};

}  // namespace generic::beta0

#include "inline/generic_players/beta0/Player.inl"
