#pragma once

#include <mutex>
#include <unordered_map>

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
    int num_threads = 1;

    bool can_reuse_subtree() const { return dirichlet_mult == 0; }
  };

  struct Results {
    GlobalPolicyCountDistr counts;
    GlobalPolicyProbDistr policy_prior;
    ValueProbDistr win_rates;
    ValueProbDistr value_prior;
  };

private:
  struct NNEvaluation {
    NNEvaluation(const NeuralNet& net, const Tensorizor& tensorizor, const GameState& state,
                 common::NeuralNet::input_vec_t& input_vec, symmetry_index_t, float inv_temp);

    LocalPolicyProbDistr local_policy_prob_distr;
    ValueProbDistr value_prob_distr;
  };

  /*
   * A Node consists of 3 main groups of non-const member variables:
   *
   * CHILDREN DATA: the addresses/number of children nodes, needed for tree traversal
   * NEURAL NETWORK EVALUATION: policy/value vectors that come from neural net evaluation
   * STATS: values that get updated throughout MCTS via backpropagation
   *
   * Of these 3, only STATS are continuously changing. The others are written only once. They are non-const in the
   * sense that they are lazily written, after-object-construction.
   *
   * During MCTS, multiple search threads will try to read and write these values. The MCTS literature is filled with
   * approaches on how to minimize thread contention, including various "lockfree" approaches that tolerate various
   * race conditions.
   *
   * See for example this 2009 paper: https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf
   *
   * For now, I am achieving thread-safety through brute-force, caveman-style. That is, all reads/writes to all Node
   * members will be protected by a single per-Node mutex. Once we have appropriate tooling to profile performance and
   * detect bottlenecks, we can improve this implementation.
   */
  class Node {
  public:
    Node(const GameState& state, const Result& result, Node* parent=nullptr, action_index_t action=-1);
    Node(const Node& node, bool prune_parent=false);

    /*
     * Releases the memory occupied by this and by all descendents, EXCEPT for the descendents of
     * protected_child (which is guaranteed to be an immediate child of this if non-null). Note that the memory of
     * protected_child itself IS released; only the *descendents* of protected_child are protected.
     *
     * In the current implementation, this works by calling delete and delete[] and by recursing down the tree.
     *
     * In future implementations, if we have object pools, this might work by releasing to an object pool.
     *
     * Also, in the future, we might have Monte Carlo *Graph* Search (MCGS) instead of MCTS. In this future, a given
     * Node might have multiple parents, so release() might decrement smart-pointer reference counts instead.
     */
    void release(Node* protected_child=nullptr);

    GlobalPolicyCountDistr get_effective_counts() const;
    void expand_children();
    void backprop(const ValueProbDistr& result, bool terminal=false);
    void terminal_backprop(const ValueProbDistr& result);

    Node* parent() const { return stable_data_.parent_; }
    int effective_count() const { return stats_.eliminated_ ? 0 : stats_.count_; }
    bool is_root() const { return !stable_data_.parent_; }
    bool has_children() const { return children_data_.num_children_; }
    bool is_leaf() const { return !has_children(); }
    bool eliminated() const { return stats_.eliminated_; }
    Node* find_child(action_index_t action) const;

    /*
     * This includes certain wins/losses AND certain draws.
     */
    bool has_certain_outcome() const { return stats_.V_floor_.sum() == 1; }

    /*
     * We only eliminate won or lost positions, not drawn positions.
     *
     * Drawn positions are not eliminated because MCTS still needs some way to compare a provably-drawn position vs an
     * uncertain position. It needs to accumulate visits to provably-drawn positions to do this.
     */
    bool can_be_eliminated() const { return stats_.V_floor_.maxCoeff() == 1; }

  private:
    float get_max_V_floor_among_children(player_index_t p) const;
    float get_min_V_floor_among_children(player_index_t p) const;

    bool is_terminal() const { return is_terminal_result(stable_data_.result_); }

    struct stable_data_t {
      stable_data_t(const GameState& state, const Result& result, Node* parent, action_index_t action);
      stable_data_t(const stable_data_t& data, bool prune_parent);

      GameState state_;
      Result result_;
      ActionMask valid_action_mask_;
      Node* parent_;
      action_index_t action_;
      player_index_t current_player_;  // is this needed?
    };

    struct children_data_t {
      Node* first_child_ = nullptr;
      int num_children_ = 0;
    };

    struct stats_t {
      stats_t();

      Eigen::Vector<float, kNumPlayers> value_avg_;
      Eigen::Vector<float, kNumPlayers> effective_value_avg_;
      Eigen::Vector<float, kNumPlayers> V_floor_;
      int count_ = 0;
      bool eliminated_ = false;
    };

    mutable std::mutex mutex_;
    const stable_data_t stable_data_;
    children_data_t children_data_;
    NNEvaluation* evaluation_ = nullptr;
    stats_t stats_;
  };

  class SearchThread {
  public:
    SearchThread(Mcts* mcts, int thread_id);
    void run();

  private:
    Mcts* const mcts_;
    const int thread_id_;
  };

  class NNEvaluationThread {
  public:
    NNEvaluationThread(int batch_size, int64_t timeout_ns);
    NNEvaluation* evaluate(const Tensorizor& tensorizor, const GameState& state, symmetry_index_t index);

  private:
    using cache_key_t = StateSymmetryIndex<GameState>;
    using cache_t = std::unordered_map<cache_key_t, NNEvaluation*>;  // TODO: use LRU cache or similar

    common::NeuralNet::input_vec_t input_vec_;
    cache_t cache_;
    const int64_t timeout_ns_;
  };

public:
  Mcts(NeuralNet& net);
  void clear();
  void receive_state_change(player_index_t, const GameState&, action_index_t, const Result&);
  const Results* sim(const Tensorizor& tensorizor, const GameState& game_state, const Params& params);
  void visit(Node*, const Tensorizor&, const GameState&, const Params&, int depth);

  static void run_search(Mcts* mcts, int thread_id);

private:
  NNEvaluationThread* nn_eval_thread_;
  NeuralNet& net_;
  Node* root_ = nullptr;
  torch::Tensor torch_input_gpu_;
  Results results_;
  player_index_t player_index_ = -1;
};

}  // namespace common

#include <common/inl/Mcts.inl>
