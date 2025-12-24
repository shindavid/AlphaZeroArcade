#pragma once

#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/Constants.hpp"
#include "core/StateChangeUpdate.hpp"
#include "generic_players/alpha0/VerboseData.hpp"
#include "generic_players/x0/Player.hpp"
#include "search/concepts/TraitsConcept.hpp"

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
    bool verbose = false;
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
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;

  using SharedData_sptr = Base::SharedData_sptr;

  Player(const Params&, SharedData_sptr, bool owns_shared_data);
  ~Player();

  void receive_state_change(const StateChangeUpdate&) override;

 protected:
  // This is virtual so that it can be overridden in tests and in DataExportingPlayer.
  virtual core::ActionResponse get_action_response_helper(const SearchResults*,
                                                          const ActionRequest&) override;

  virtual PolicyTensor get_action_policy(const SearchResults*, const ActionMask&) const override;

  void apply_LCB(const SearchResults* mcts_results, const ActionMask&, PolicyTensor& policy) const;

  const ParamsExtra params_extra_;
  VerboseData<Traits>* verbose_info_ = nullptr;

  template <core::concepts::EvalSpec ES>
  friend class PlayerTest;
};

}  // namespace generic::alpha0

#include "inline/generic_players/alpha0/Player.inl"
