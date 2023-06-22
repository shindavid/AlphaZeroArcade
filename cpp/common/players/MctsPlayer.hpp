#pragma once

#include <ostream>

#include <boost/filesystem.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Mcts.hpp>
#include <common/MctsResults.hpp>
#include <common/TensorizorConcept.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/Math.hpp>

namespace common {

/*
 * The MctsPlayer uses MCTS to select actions.
 *
 * Note that when 2 or more identically-configured MctsPlayer's are playing in the same game, they can share the same
 * MCTS tree, as an optimization. This implementation supports this optimization.
 */
template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class MctsPlayer : public AbstractPlayer<GameState_> {
public:
  using base_t = AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;

  // See KataGo paper for description of search modes.
  enum SearchMode {
    kFast,
    kFull,
    kRawPolicy,
    kNumSearchModes
  };

  enum DefaultParamsType {
    kCompetitive,
    kTraining
  };

  struct Params {
    Params(DefaultParamsType);
    void dump() const;

    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    std::string move_temperature_str;
    int num_raw_policy_starting_moves = 0;
    bool verbose = false;
  };

  using GameStateTypes = common::GameStateTypes<GameState>;

  using dtype = typename GameStateTypes::dtype;
  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsSearchParams = typename Mcts::SearchParams;
  using MctsResults = common::MctsResults<GameState>;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;

  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ValueArray = typename Mcts::ValueArray;
  using LocalPolicyArray = typename Mcts::LocalPolicyArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using PolicyShape = typename GameStateTypes::PolicyShape;
  using PolicyArray = typename GameStateTypes::PolicyArray;

  MctsPlayer(const Params&, Mcts* mcts);  // uses this constructor when sharing an MCTS tree
  template <typename... Ts> MctsPlayer(const Params&, Ts&&... mcts_params_args);
  ~MctsPlayer();

  Mcts* get_mcts() { return mcts_; }
  void start_game() override;
  void receive_state_change(seat_index_t, const GameState&, action_index_t) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;
  void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;
  void set_facing_human_tui_player() override { facing_human_tui_player_ = true; }  // affects printing

protected:
  const MctsResults* mcts_search(const GameState& state, SearchMode search_mode) const;
  SearchMode choose_search_mode() const;
  action_index_t get_action_helper(SearchMode, const MctsResults*, const ActionMask& valid_actions) const;

  struct VerboseInfo {
    LocalPolicyArray action_policy;
    MctsResults mcts_results;

    bool initialized = false;
  };

  SearchMode get_random_search_mode() const;
  void verbose_dump() const;

  const Params params_;
  Tensorizor tensorizor_;

  Mcts* mcts_;
  const MctsSearchParams search_params_[kNumSearchModes];
  math::ExponentialDecay move_temperature_;
  VerboseInfo* verbose_info_ = nullptr;
  bool owns_mcts_;
  bool facing_human_tui_player_ = false;
  int move_count_ = 0;
};

}  // namespace common

#include <common/players/inl/MctsPlayer.inl>

