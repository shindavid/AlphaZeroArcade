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
#include <util/EigenTorch.hpp>
#include <util/Math.hpp>

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class MctsPlayer : public AbstractPlayer<GameState_> {
public:
  using base_t = AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;

  enum SimType {
    kFast,
    kFull,
    kRawPolicy,
    kNumSimTypes
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

  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsParams = typename Mcts::Params;
  using MctsSimParams = typename Mcts::SimParams;
  using MctsResults = common::MctsResults<GameState>;

  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using player_array_t = typename base_t::player_array_t;
  using ValueProbDistr = typename Mcts::ValueProbDistr;
  using LocalPolicyProbDistr = typename Mcts::LocalPolicyProbDistr;
  using GlobalPolicyProbDistr = typename GameStateTypes::GlobalPolicyProbDistr;

  MctsPlayer(const Params&, Mcts* mcts);
  template <typename... Ts> MctsPlayer(const Params&, Ts&&... mcts_params_args);
  ~MctsPlayer();

  Mcts* get_mcts() { return mcts_; }
  void start_game(game_id_t, const player_array_t& players, player_index_t seat_assignment) override;
  void receive_state_change(player_index_t, const GameState&, action_index_t, const GameOutcome&) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;
  void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;
  float avg_batch_size() const { return mcts_->avg_batch_size(); }

protected:
  const MctsResults* mcts_sim(const GameState& state, SimType sim_type) const;
  SimType choose_sim_type() const;
  action_index_t get_action_helper(SimType, const MctsResults*, const ActionMask& valid_actions) const;

  struct VerboseInfo {
    ValueProbDistr mcts_value;
    LocalPolicyProbDistr mcts_policy;
    MctsResults mcts_results;

    bool initialized = false;
  };

  SimType get_random_sim_type() const;
  void verbose_dump() const;

  const Params params_;
  Tensorizor tensorizor_;

  Mcts* mcts_;
  const MctsSimParams sim_params_[kNumSimTypes];
  math::ExponentialDecay move_temperature_;
  player_index_t my_index_ = -1;
  VerboseInfo* verbose_info_ = nullptr;
  bool owns_mcts_;
  bool facing_human_tui_player_ = false;
  int move_count_ = 0;
};

}  // namespace common

#include <common/inl/MctsPlayer.inl>
