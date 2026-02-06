#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/ActionRequest.hpp"
#include "core/ActionResponse.hpp"
#include "core/Constants.hpp"
#include "core/StateChangeUpdate.hpp"
#include "core/StateIterator.hpp"
#include "search/AuxData.hpp"
#include "search/Constants.hpp"
#include "search/Manager.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchResponse.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <memory>

namespace generic::x0 {

/*
 * A base-class for {alpha0,beta0}::Player.
 *
 * Note that when 2 or more identically-configured generic::x0::Player's are playing in the same
 * game, they can share the same MCTS tree, as an optimization. This implementation supports this
 * optimization.
 */
template <search::concepts::Traits Traits_>
class Player : public core::AbstractPlayer<typename Traits_::Game> {
 public:
  using Traits = Traits_;
  using Game = Traits::Game;
  using EvalSpec = Traits::EvalSpec;

  struct Params {
    Params(search::Mode);

    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    float starting_move_temperature;
    float ending_move_temperature = 0.2;
    float move_temperature_half_life = 0.5 * EvalSpec::MctsConfiguration::kOpeningLength;
    bool verbose = false;
  };

  using Manager = search::Manager<Traits>;
  using SearchResults = Traits::SearchResults;
  using SearchResponse = search::SearchResponse<SearchResults>;

  using State = Game::State;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = core::ActionRequest<Game>;
  using PolicyTensor = Game::Types::PolicyTensor;
  using StateChangeUpdate = core::StateChangeUpdate<Game>;
  using StateIterator = core::StateIterator<Game>;
  using AuxData = search::AuxData<Traits>;
  using GameResultTensor = Game::GameResults::Tensor;

  struct SharedData {
    template <typename... Ts>
    SharedData(Ts&&... args) : manager(std::forward<Ts>(args)...) {}

    Manager manager;
    int num_raw_policy_starting_moves = 0;
  };
  using SharedData_sptr = std::shared_ptr<SharedData>;

  Player(const Params&, SharedData_sptr, bool owns_shared_data);

  Manager* get_manager() const { return &shared_data_->manager; }
  bool start_game() override;
  void receive_state_change(const StateChangeUpdate&) override;
  core::ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State& state, const GameResultTensor& results) override;

 protected:
  void clear_search_mode();
  void init_search_mode(const ActionRequest&);

  virtual core::ActionResponse get_action_response_helper(const SearchResults*,
                                                          const ActionRequest&);

  virtual PolicyTensor get_action_policy(const SearchResults*, const ActionMask&) const = 0;

  void raw_init(const SearchResults*, const ActionMask&, PolicyTensor& policy) const;
  void apply_temperature(PolicyTensor& policy) const;
  void normalize(const ActionMask&, PolicyTensor& policy) const;

  core::SearchMode get_random_search_mode() const;
  bool verbose() const { return params_.verbose; }

  const Params params_;

  const search::SearchParams search_params_[core::kNumSearchModes];
  math::ExponentialDecay move_temperature_;
  SharedData_sptr shared_data_;
  const bool owns_shared_data_;

  mutable mit::mutex search_mode_mutex_;
  core::SearchMode search_mode_ = core::kNumSearchModes;
  std::vector<AuxData*> aux_data_ptrs_;
};

}  // namespace generic::x0

#include "inline/generic_players/x0/Player.inl"
