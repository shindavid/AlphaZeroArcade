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

namespace common {

template<GameStateConcept GameState_, TensorizorConcept<GameState_> Tensorizor_>
class MctsPlayer : public AbstractPlayer<GameState_> {
public:
  using base_t = AbstractPlayer<GameState_>;
  using GameState = GameState_;
  using Tensorizor = Tensorizor_;

  enum SimType {
    kFast,
    kFull
  };

  enum DefaultParamsType {
    kCompetitive,
    kTraining
  };

  struct Params {
    Params(DefaultParamsType);
    void dump() const;

    template<boost_util::program_options::OptionStyle=boost_util::program_options::kUseAbbreviations>
    auto make_options_description();

    int num_fast_iters;
    int num_full_iters;
    float full_pct;
    float temperature;
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
  const MctsSimParams fast_sim_params_;
  const MctsSimParams full_sim_params_;
  const float inv_temperature_;
  player_index_t my_index_ = -1;
  VerboseInfo* verbose_info_ = nullptr;
  bool owns_mcts_;

  // below are pseudo-return values of get_action(), to be used by derived class
  const MctsResults* mcts_results_;
  SimType sim_type_;
};

}  // namespace common

#include <common/inl/MctsPlayer.inl>
