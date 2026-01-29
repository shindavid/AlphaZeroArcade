#pragma once

#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/Constants.hpp"
#include "core/StateChangeUpdate.hpp"
#include "generic_players/x0/Player.hpp"
#include "search/AuxData.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <vector>

namespace generic::alpha0 {

/*
 * The generic::alpha0::Player uses AlphaZero MCTS to select actions.
 */
template <search::concepts::Traits Traits_>
class Player : public generic::x0::Player<Traits_> {
 public:
  using Base = generic::x0::Player<Traits_>;
  using BasePlayer = Player;  // a little ugly, but needed for generic::x0::PlayerGeneratorBase
  using Traits = Traits_;
  using Game = Traits::Game;
  using EvalSpec = Traits::EvalSpec;
  using BaseParams = Base::Params;

  struct ParamsExtra {
    float LCB_z_score = 2.0;
    int verbose_num_rows_to_display = core::kNumRowsToDisplayVerbose;
  };

  struct Params : public BaseParams, public ParamsExtra {
    using BaseParams::BaseParams;

    auto make_options_description();
  };

  using SearchResults = Traits::SearchResults;

  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = core::ActionRequest<Game>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using GameResultTensor = Game::GameResults::Tensor;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using VerboseData = Traits::VerboseData;
  using AuxData = search::AuxData<Traits>;

  using SharedData_sptr = Base::SharedData_sptr;

  Player(const Params& params, SharedData_sptr shared_data, bool owns_shared_data)
      : Base(params, shared_data, owns_shared_data), params_extra_(params) {}

  void receive_state_change(const StateChangeUpdate&) override;

 protected:
  // This is virtual so that it can be overridden in tests and in DataExportingPlayer.
  virtual core::ActionResponse get_action_response_helper(const SearchResults*,
                                                          const ActionRequest&) override;

  virtual PolicyTensor get_action_policy(const SearchResults*, const ActionMask&) const override;

  void apply_LCB(const SearchResults* mcts_results, const ActionMask&, PolicyTensor& policy) const;

  const ParamsExtra params_extra_;

  template <core::concepts::EvalSpec ES>
  friend class PlayerTest;
};

}  // namespace generic::alpha0

#include "inline/generic_players/alpha0/Player.inl"
