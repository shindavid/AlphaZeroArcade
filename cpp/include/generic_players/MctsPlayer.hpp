#pragma once

#include <ostream>

#include <boost/filesystem.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <core/AbstractPlayer.hpp>
#include <core/BasicTypes.hpp>
#include <core/Constants.hpp>
#include <core/concepts/Game.hpp>
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
template <core::concepts::Game Game>
class MctsPlayer : public core::AbstractPlayer<Game> {
 public:
  using base_t = core::AbstractPlayer<Game>;

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

  using MctsManager = mcts::Manager<Game>;
  using MctsSearchParams = mcts::SearchParams;
  using MctsSearchResults = mcts::SearchResults<Game>;
  using player_name_array_t = typename Game::player_name_array_t;

  using FullState = typename Game::FullState;
  using IO = typename Game::IO;
  using ActionMask = typename Game::ActionMask;
  using ValueArray = typename Game::ValueArray;
  using PolicyTensor = typename Game::PolicyTensor;

  // uses this constructor when sharing an MCTS manager
  MctsPlayer(const Params&, MctsManager* mcts_manager);
  template <typename... Ts>
  MctsPlayer(const Params&, Ts&&... mcts_params_args);
  ~MctsPlayer();

  MctsManager* get_mcts_manager() { return mcts_manager_; }
  void start_game() override;
  void receive_state_change(core::seat_index_t, const FullState&, core::action_t) override;
  core::ActionResponse get_action_response(const FullState&, const ActionMask&) override;
  void set_facing_human_tui_player() override {
    facing_human_tui_player_ = true;  // affects printing
  }

 protected:
  const MctsSearchResults* mcts_search(const FullState& state, core::SearchMode search_mode) const;
  core::SearchMode choose_search_mode() const;
  core::ActionResponse get_action_response_helper(core::SearchMode, const MctsSearchResults*,
                                                  const ActionMask& valid_actions) const;

  struct VerboseInfo {
    PolicyTensor action_policy;
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
