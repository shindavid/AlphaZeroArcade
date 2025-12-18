#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/Constants.hpp"
#include "generic_players/alpha0/VerboseData.hpp"
#include "search/AlgorithmsFor.hpp"
#include "search/Constants.hpp"
#include "search/Manager.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchResponse.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <memory>

namespace generic::alpha0 {

/*
 * The generic::alpha0::Player uses AlphaZero MCTS to select actions.
 *
 * Note that when 2 or more identically-configured generic::alpha0::Player's are playing in the same
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
    void dump() const;

    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    float starting_move_temperature;
    float ending_move_temperature = 0.2;
    float move_temperature_half_life = 0.5 * EvalSpec::MctsConfiguration::kOpeningLength;
    float LCB_z_score = 2.0;
    bool verbose = false;
    int verbose_num_rows_to_display = core::kNumRowsToDisplayVerbose;
  };

  using Algorithms = search::AlgorithmsForT<Traits>;
  using Manager = search::Manager<Traits>;
  using SearchResults = Traits::SearchResults;
  using SearchResponse = search::SearchResponse<SearchResults>;
  using player_name_array_t = Game::Types::player_name_array_t;

  using State = Game::State;
  using IO = Game::IO;
  using Rules = Game::Rules;
  using Constants = Game::Constants;
  using ActionMask = Game::Types::ActionMask;
  using ActionRequest = Game::Types::ActionRequest;
  using ActionResponse = Game::Types::ActionResponse;
  using ValueArray = Game::Types::ValueArray;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using GameResultTensor = Game::Types::GameResultTensor;
  using StateChangeUpdate = Game::Types::StateChangeUpdate;

  struct SharedData {
    template <typename... Ts>
    SharedData(Ts&&... args) : manager(std::forward<Ts>(args)...) {}

    Manager manager;
    int num_raw_policy_starting_moves = 0;
  };
  using SharedData_sptr = std::shared_ptr<SharedData>;

  Player(const Params&, SharedData_sptr, bool owns_shared_data);
  ~Player();

  Manager* get_manager() const { return &shared_data_->manager; }
  bool start_game() override;
  void receive_state_change(const StateChangeUpdate&) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void end_game(const State&, const GameResultTensor&) override;

 protected:
  void clear_search_mode();
  void init_search_mode(const ActionRequest&);

  // This is virtual so that it can be overridden in tests and in DataExportingPlayer.
  virtual ActionResponse get_action_response_helper(const SearchResults*, const ActionRequest&);

  auto get_action_policy(const SearchResults*, const ActionMask&) const;

  void raw_init(const SearchResults*, const ActionMask&, PolicyTensor& policy) const;
  void apply_temperature(PolicyTensor& policy) const;
  void apply_LCB(const SearchResults* mcts_results, const ActionMask&, PolicyTensor& policy) const;
  void normalize(const ActionMask&, PolicyTensor& policy) const;

  core::SearchMode get_random_search_mode() const;

  const Params params_;

  const search::SearchParams search_params_[core::kNumSearchModes];
  math::ExponentialDecay move_temperature_;
  SharedData_sptr shared_data_;
  VerboseData<Traits>* verbose_info_ = nullptr;
  const bool owns_shared_data_;
  std::vector<const SearchResults*> search_result_ptrs_;

  mutable mit::mutex search_mode_mutex_;
  core::SearchMode search_mode_ = core::kNumSearchModes;

  template <core::concepts::EvalSpec ES>
  friend class PlayerTest;
};

}  // namespace generic::alpha0

#include "inline/generic_players/alpha0/Player.inl"
