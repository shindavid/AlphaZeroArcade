#pragma once

#include "core/ActionPrinter.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "generic_players/x0/Player.hpp"
#include "search/AuxData.hpp"
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
  using Move = Game::Move;
  using MoveList = Game::MoveList;
  using BaseParams = Base::Params;
  using ActionPrinter = core::ActionPrinter<Game>;

  struct ParamsExtra {
    float LCB_z_score = 2.0;
    int verbose_num_rows_to_display = core::kNumRowsToDisplayVerbose;
  };

  struct Params : public BaseParams, public ParamsExtra {
    using BaseParams::BaseParams;

    auto make_options_description();
  };

  using SharedData_sptr = Base::SharedData_sptr;
  using SearchResults = Traits::SearchResults;
  using PolicyTensor = Game::Types::PolicyTensor;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using ActionRequest = core::ActionRequest<Game>;
  using ActionResponse = core::ActionResponse<Game>;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using VerboseData = Traits::VerboseData;
  using State = Game::State;
  using AuxData = search::AuxData<Traits>;
  using PolicyEncoding = Traits::EvalSpec::PolicyEncoding;

  Player(const Params& params, SharedData_sptr shared_data, bool owns_shared_data)
      : Base(params, shared_data, owns_shared_data), params_extra_(params) {}

  void receive_state_change(const StateChangeUpdate&) override;

 protected:
  virtual ActionResponse get_action_response_helper(const SearchResults*,
                                                    const ActionRequest&) override;
  virtual PolicyTensor get_action_policy(const SearchResults*, const MoveList&) const override;

  void apply_LCB(const SearchResults* mcts_results, const MoveList&, PolicyTensor& policy) const;

  const ParamsExtra params_extra_;
};

}  // namespace generic::beta0

#include "inline/generic_players/beta0/Player.inl"
