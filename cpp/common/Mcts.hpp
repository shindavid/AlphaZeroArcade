#pragma once

#include <Eigen/Core>

#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/NeuralNet.hpp>
#include <common/TensorizorConcept.hpp>
#include <common/BasicTypes.hpp>
#include <util/BitSet.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class Mcts {
public:
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameState::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using GlobalPolicyCountDistr = Eigen::Vector<int, kNumGlobalActions>;
  using GlobalPolicyProbDistr = Eigen::Vector<float, kNumGlobalActions>;
  using ValueProbDistr = Eigen::Vector<float, kNumPlayers>;
  using Result = typename GameStateTypes<GameState>::Result;
  using ActionMask = util::BitSet<kNumGlobalActions>;
  using LocalPolicyLogitDistr = Eigen::Matrix<float, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;
  using LocalPolicyProbDistr = Eigen::Matrix<float, Eigen::Dynamic, 1, 0, kMaxNumLocalActions>;

  struct Params {
    int tree_size_limit;
    float root_softmax_temperature;
    float cPUCT = 1.1;
    float dirichlet_mult = 0.25;
    float dirichlet_alpha = 0.03;
    bool allow_eliminations = true;

    bool can_reuse_subtree() const { return dirichlet_mult == 0; }
  };

  struct Results {
    GlobalPolicyCountDistr counts;
    GlobalPolicyProbDistr policy_prior;
    ValueProbDistr win_rates;
    ValueProbDistr value_prior;
  };

private:
  struct StateEvaluation {
    void init(const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state, const Result& result,
              common::NeuralNet::input_vec_t& input_vec, float inv_temp);
    bool is_terminal() const { return is_terminal_result(result); }

    player_index_t current_player;
    Result result;

    // Below members are only valid if !is_terminal()
    ActionMask valid_action_mask;
    LocalPolicyProbDistr local_policy_prob_distr;
    ValueProbDistr value_prob_distr;
    bool initialized = false;
  };

  class Tree {
  public:
    Tree(action_index_t action=-1, Tree* parent=nullptr);

    GlobalPolicyCountDistr get_effective_counts() const;
    void expand_children();
    void backprop(const ValueProbDistr& result, bool terminal=false);
    void terminal_backprop(const ValueProbDistr& result);

    int effective_count() const { return eliminated_ ? 0 : count_; }
    bool is_root() const { return !parent_; }
    bool has_children() const { return num_children_; }
    bool is_leaf() const { return !has_children(); }

    /*
     * This includes certain wins/losses AND certain draws.
     */
    bool has_certain_outcome() const { return V_floor_.sum() == 1; }

    /*
     * We only eliminate won or lost positions, not drawn positions.
     *
     * Drawn positions are not eliminated because MCTS still needs some way to compare a provably-drawn position vs an
     * uncertain position. It needs to accumulate visits to provably-drawn positions to do this.
     */
    bool can_be_eliminated() const { return V_floor_.maxCoeff() == 1; }

  private:
    float get_max_V_floor_among_children(player_index_t p) const;
    float get_min_V_floor_among_children(player_index_t p) const;

    StateEvaluation evaluation_;  // only valid if evaluated_
    Tree* parent_;
    Tree** children_ = nullptr;
    int num_children_ = 0;
    action_index_t action_;
    int count_ = 0;
    Eigen::Vector<float, kNumPlayers> value_sum_;
    Eigen::Vector<float, kNumPlayers> value_avg_;
    Eigen::Vector<float, kNumPlayers> effective_value_avg_;
    Eigen::Vector<float, kNumPlayers> V_floor_;
    bool eliminated_ = false;
  };

public:
  Mcts();
  void clear();
  void receive_state_change(player_index_t, const GameState&, action_index_t, const Result&);
  const Results* sim(const Tensorizor& tensorizor, const GameState& game_state, const Params& params);

private:
  Tree* root_ = nullptr;
  torch::Tensor torch_input_gpu_;
  Results results_;
};

}  // namespace common

#include <common/inl/Mcts.inl>
