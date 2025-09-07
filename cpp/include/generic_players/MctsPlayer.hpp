#pragma once

#include "alphazero/ManagerParams.hpp"
#include "alphazero/SearchResults.hpp"
#include "alphazero/Traits.hpp"
#include "core/AbstractPlayer.hpp"
#include "core/BasicTypes.hpp"
#include "core/Constants.hpp"
#include "core/concepts/EvalSpecConcept.hpp"
#include "search/Constants.hpp"
#include "search/Manager.hpp"
#include "search/SearchParams.hpp"
#include "search/SearchResponse.hpp"
#include "util/Math.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <memory>

namespace generic {

/*
 * The MctsPlayer uses MCTS to select actions.
 *
 * Note that when 2 or more identically-configured MctsPlayer's are playing in the same game, they
 * can share the same MCTS tree, as an optimization. This implementation supports this optimization.
 */
template <core::concepts::EvalSpec EvalSpec>
class MctsPlayer : public core::AbstractPlayer<typename EvalSpec::Game> {
 public:
  using Game = EvalSpec::Game;
  using base_t = core::AbstractPlayer<Game>;

  struct Params {
    Params(search::Mode);
    void dump() const;

    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    float starting_move_temperature;
    float ending_move_temperature = 0.2;
    float move_temperature_half_life = 0.5 * Game::MctsConfiguration::kOpeningLength;
    float LCB_z_score = 2.0;
    bool verbose = false;
    int verbose_num_rows_to_display = core::kNumRowsToDisplayVerbose;
  };

  using Traits = alpha0::Traits<Game>;
  using MctsManager = search::Manager<Traits>;
  using MctsManagerParams = alpha0::ManagerParams<EvalSpec>;
  using SearchResults = alpha0::SearchResults<Game>;
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

  struct SharedData {
    template <typename... Ts>
    SharedData(Ts&&... args) : manager(std::forward<Ts>(args)...) {}

    MctsManager manager;
    int num_raw_policy_starting_moves = 0;
  };
  using SharedData_sptr = std::shared_ptr<SharedData>;

  MctsPlayer(const Params&, SharedData_sptr, bool owns_shared_data);
  ~MctsPlayer();

  MctsManager* get_manager() const { return &shared_data_->manager; }
  void start_game() override;
  void receive_state_change(core::seat_index_t, const State&, core::action_t) override;
  ActionResponse get_action_response(const ActionRequest&) override;
  void set_facing_human_tui_player() override {
    facing_human_tui_player_ = true;  // affects printing
  }

 protected:
  void clear_search_mode();

  // assumes search_mode_mutex_ is held, returns true if search mode was changed
  bool init_search_mode(const ActionRequest&);

  auto get_action_policy(const SearchResults*, const ActionMask&) const;

  // This is virtual so that it can be overridden in tests.
  virtual ActionResponse get_action_response_helper(const SearchResults*,
                                                    const ActionMask& valid_actions) const;

  void print_mcts_results(std::ostream& ss, const PolicyTensor& action_policy,
                          const SearchResults& results) const;

  struct VerboseInfo {
    PolicyTensor action_policy;
    SearchResults mcts_results;

    bool initialized = false;
  };

  core::SearchMode get_random_search_mode() const;
  void verbose_dump() const;

  const Params params_;

  const search::SearchParams search_params_[core::kNumSearchModes];
  math::ExponentialDecay move_temperature_;
  SharedData_sptr shared_data_;
  VerboseInfo* verbose_info_ = nullptr;
  const bool owns_shared_data_;
  bool facing_human_tui_player_ = false;

  mutable mit::mutex search_mode_mutex_;
  core::SearchMode search_mode_ = core::kNumSearchModes;

  template <core::concepts::Game>
  friend class MctsPlayerTest;
};

}  // namespace generic

#include "inline/generic_players/MctsPlayer.inl"
