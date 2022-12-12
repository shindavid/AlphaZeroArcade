#pragma once

#include <mutex>
#include <unordered_map>

#include <Eigen/Core>
#include <EigenRand/EigenRand>

#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/MctsResults.hpp>
#include <common/NeuralNet.hpp>
#include <common/TensorizorConcept.hpp>
#include <util/BitSet.hpp>
#include <util/EigenTorch.hpp>
#include <util/LRUCache.hpp>

namespace common {

/*
 * TODO: move the various inner-classes of Mcts_ into separate files as standalone-classes.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class Mcts_ {
public:
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameState::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using TensorizorTypes = TensorizorTypes_<Tensorizor>;
  using GameStateTypes = GameStateTypes_<GameState>;

  using MctsResults = MctsResults_<GameState>;
  using ValueProbDistr = typename GameStateTypes::ValueProbDistr;
  using GameResult = typename GameStateTypes::GameResult;
  using ActionMask = util::BitSet<kNumGlobalActions>;
  using LocalPolicyProbDistr = typename GameStateTypes::LocalPolicyProbDistr;
  using LocalPolicyCountDistr = typename GameStateTypes::LocalPolicyCountDistr;

  using FullInputTensor = typename TensorizorTypes::DynamicInputTensor;
  using FullValueMatrix = typename GameStateTypes::template ValueMatrix<Eigen::Dynamic>;
  using FullPolicyMatrix = typename GameStateTypes::template PolicyMatrix<Eigen::Dynamic>;
  using ValueVector = typename GameStateTypes::ValueVector;
  using PolicyVector = typename GameStateTypes::PolicyVector;

  struct Params {
    int tree_size_limit = 100;
    float root_softmax_temperature = 1.03;
    float cPUCT = 1.1;
    float dirichlet_mult = 0.25;
    float dirichlet_alpha = 0.03;
    bool allow_eliminations = true;
    int num_threads = 1;

    bool can_reuse_subtree() const { return dirichlet_mult == 0; }
  };

private:
  class NNEvaluation {
  public:
    NNEvaluation(const ValueVector& value, const PolicyVector& policy, const ActionMask& valid_actions, float inv_temp);
    const ValueProbDistr& value_prob_distr() const { return value_prob_distr_; }
    const LocalPolicyProbDistr& local_policy_prob_distr() const { return local_policy_prob_distr_; }

  protected:
    ValueProbDistr value_prob_distr_;
    LocalPolicyProbDistr local_policy_prob_distr_;
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
   * For now, I am achieving thread-safety by having three mutexes per-Node, one for each of the above three
   * categories. Once we have appropriate tooling to profile performance and detect bottlenecks, we can improve this
   * implementation.
   *
   * NAMING NOTE: Methods with a leading underscore are NOT thread-safe. Such methods are expected to be called in
   * a context that guarantees the appropriate level of thread-safety.
   */
  class Node {
  public:
    Node(const Tensorizor& tensorizor, const GameState& state, const GameResult& result, symmetry_index_t sym_index,
         Node* parent=nullptr, action_index_t action=-1);
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
    void _release(Node* protected_child=nullptr);

    std::mutex& evaluation_mutex() { return evaluation_mutex_; }
    std::mutex& stats_mutex() { return stats_mutex_; }

    LocalPolicyCountDistr get_effective_counts() const;
    bool expand_children();  // returns false iff already has children
    void backprop(const ValueProbDistr& result, bool terminal=false);
    void terminal_backprop(const ValueProbDistr& result);

    const Tensorizor& tensorizor() const { return stable_data_.tensorizor_; }
    const GameState& state() const { return stable_data_.state_; }
    const GameResult& result() const { return stable_data_.result_; }
    action_index_t action() const { return stable_data_.action_; }
    Node* parent() const { return stable_data_.parent_; }
    bool is_root() const { return !stable_data_.parent_; }
    symmetry_index_t sym_index() const { return stable_data_.sym_index_; }
    player_index_t current_player() const { return stable_data_.current_player_; }
    bool is_terminal() const { return is_terminal_result(stable_data_.result_); }

    bool _has_children() const { return children_data_.num_children_; }
    int _num_children() const { return children_data_.num_children_; }
    Node* _get_child(int c) const { return children_data_.first_child_ + c; }
    Node* _find_child(action_index_t action) const;

    bool _eliminated() const { return stats_.eliminated_; }
    float _V_floor(player_index_t p) const { return stats_.V_floor_(p); }
    float _effective_value_avg(player_index_t p) const { return stats_.effective_value_avg_(p); }
    int _effective_count() const { return stats_.eliminated_ ? 0 : stats_.count_; }
    bool _has_certain_outcome() const { return stats_.V_floor_.sum() == 1; }  // won, lost, OR drawn positions
    bool _can_be_eliminated() const { return stats_.V_floor_.maxCoeff() == 1; }  // won/lost positions, not drawn ones

    NNEvaluation* _evaluation() const { return evaluation_; }
    NNEvaluation** _evaluation_ptr() { return &evaluation_; }

  private:
    float _get_max_V_floor_among_children(player_index_t p) const;
    float _get_min_V_floor_among_children(player_index_t p) const;

    struct stable_data_t {
      stable_data_t(const Tensorizor& tensorizor, const GameState& state, const GameResult& result, Node* parent,
                    symmetry_index_t sym_index, action_index_t action);
      stable_data_t(const stable_data_t& data, bool prune_parent);

      Tensorizor tensorizor_;
      GameState state_;
      GameResult result_;
      ActionMask valid_action_mask_;
      Node* parent_;
      symmetry_index_t sym_index_;
      action_index_t action_;
      player_index_t current_player_;
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

    mutable std::mutex children_data_mutex_;
    mutable std::mutex evaluation_mutex_;
    mutable std::mutex stats_mutex_;
    const stable_data_t stable_data_;
    children_data_t children_data_;
    NNEvaluation* evaluation_ = nullptr;  // TODO: use smart-pointer
    stats_t stats_;
  };

  class SearchThread {
  public:
    SearchThread(Mcts_* mcts, int thread_id);
    void run();

  private:
    Mcts_* const mcts_;
    const int thread_id_;
  };

  class NNEvaluationThread {
  public:
    NNEvaluationThread(NeuralNet& net, int batch_size, int64_t timeout_ns, int cache_size);
    ~NNEvaluationThread() { delete[] evaluation_data_arr_; }
    void evaluate(const Tensorizor& tensorizor, const GameState& state, symmetry_index_t sym_index, float inv_temp,
                  NNEvaluation** eval_ptr);

  private:
    using cache_key_t = StateEvaluationKey<GameState>;
    using cache_t = util::LRUCache<cache_key_t, NNEvaluation*>;
    using evaluation_pool_t = std::vector<NNEvaluation>;  // TODO: use smart-pointer-compatible object-pool

    struct evaluation_data_t {
      NNEvaluation** eval_ptr;
      cache_key_t cache_key;
      ActionMask valid_actions;
    };

    NeuralNet& net_;
    FullPolicyMatrix policy_batch_;
    FullValueMatrix value_batch_;
    FullInputTensor input_batch_;
    evaluation_data_t* evaluation_data_arr_;
    evaluation_pool_t evaluation_pool_;

    common::NeuralNet::input_vec_t input_vec_;
    torch::Tensor torch_input_gpu_;

    cache_t cache_;
    const int64_t timeout_ns_;
    const int batch_size_limit_;
    int batch_write_index_ = 0;
  };

public:
  Mcts_(NeuralNet& net, int batch_size, int64_t timeout_ns, int cache_size);
  ~Mcts_();

  void clear();
  void receive_state_change(player_index_t, const GameState&, action_index_t, const GameResult&);
  const MctsResults* sim(const Tensorizor& tensorizor, const GameState& game_state, const Params& params);
  void visit(Node*, const Params&, int depth);

  static void run_search(Mcts_* mcts, int thread_id);

private:
  eigen_util::UniformDirichletGen<float> dirichlet_gen_;
  Eigen::Rand::P8_mt19937_64 rng_;

  NNEvaluationThread* nn_eval_thread_ = nullptr;
  Node* root_ = nullptr;
  torch::Tensor torch_input_gpu_;
  MctsResults results_;
  player_index_t player_index_ = -1;
};

}  // namespace common

#include <common/inl/Mcts.inl>
