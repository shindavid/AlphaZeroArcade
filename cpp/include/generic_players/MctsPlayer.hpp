#pragma once

#include <ostream>

#include <boost/filesystem.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/Constants.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/Manager.hpp>
#include <mcts/SearchParams.hpp>
#include <mcts/SearchResults.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Math.hpp>

namespace generic {

/*
 * The MctsPlayer uses MCTS to select actions.
 *
 * Note that when 2 or more identically-configured MctsPlayer's are playing in the same game, they
 * can share the same MCTS tree, as an optimization. This implementation supports this optimization.
 */
template <core::GameStateConcept GameState_, core::TensorizorConcept<GameState_> Tensorizor_>
class MctsPlayer : public core::AbstractPlayer<GameState_> {
 public:
  using base_t = core::AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;

  struct Params {
    Params(mcts::Mode);
    void dump() const;

    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    std::string move_temperature_str;
    int num_raw_policy_starting_moves = 0;
    bool verbose = false;
  };

  using GameStateTypes = core::GameStateTypes<GameState>;

  using dtype = typename GameStateTypes::dtype;
  using MctsManager = mcts::Manager<GameState, Tensorizor>;
  using MctsSearchParams = mcts::SearchParams;
  using MctsSearchResults = mcts::SearchResults<GameState>;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;

  using Action = typename GameStateTypes::Action;
  using ActionResponse = typename GameStateTypes::ActionResponse;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ValueArray = typename GameStateTypes::ValueArray;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using PolicyArray = typename GameStateTypes::PolicyArray;

  // uses this constructor when sharing an MCTS manager
  MctsPlayer(const Params&, MctsManager* mcts_manager);
  template <typename... Ts>
  MctsPlayer(const Params&, Ts&&... mcts_params_args);
  ~MctsPlayer();

  MctsManager* get_mcts_manager() { return mcts_manager_; }
  void start_game() override;
  void receive_state_change(core::seat_index_t, const GameState&, const Action&) override;
  ActionResponse get_action_response(const GameState&, const ActionMask&) override;
  void set_facing_human_tui_player() override {
    facing_human_tui_player_ = true;  // affects printing
  }

 protected:
  const MctsSearchResults* mcts_search(const GameState& state, core::SearchMode search_mode) const;
  core::SearchMode choose_search_mode() const;
  ActionResponse get_action_response_helper(core::SearchMode, const MctsSearchResults*,
                                            const ActionMask& valid_actions) const;

  struct VerboseInfo {
    LocalPolicyArray action_policy;
    MctsSearchResults mcts_results;

    bool initialized = false;
  };

  core::SearchMode get_random_search_mode() const;
  void verbose_dump() const;

  const Params params_;

  MctsManager* mcts_manager_;
  const MctsSearchParams search_params_[core::kNumSearchModes];
  math::ExponentialDecay move_temperature_;
  VerboseInfo* verbose_info_ = nullptr;
  bool owns_manager_;
  bool facing_human_tui_player_ = false;
  int move_count_ = 0;
};

}  // namespace generic

#include <inline/generic_players/MctsPlayer.inl>
