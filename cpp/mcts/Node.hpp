#pragma once

#include <cstdint>
#include <mutex>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluation.hpp>

namespace mcts {

/*
 * A Node consists of n=3 main groups of non-const member variables:
 *
 * children_data_: pointers of children nodes, needed for tree traversal
 * evaluation_data_: policy/value vectors that come from neural net evaluation
 * stats_: values that get updated throughout MCTS via backpropagation
 *
 * During MCTS, multiple search threads will try to read and write these values. Thread-safety is achieved in a
 * high-performance manner through mutexes and condition variables.
 */
template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class Node {
public:
  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using GameStateTypes = core::GameStateTypes<GameState>;

  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;
  static constexpr int kNumGlobalActions = GameStateTypes::kNumGlobalActions;
  static constexpr int kNumPlayers = GameState::kNumPlayers;

  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;

  using child_index_t = int;
  using player_bitset_t = std::bitset<kNumPlayers>;

  enum evaluation_state_t : int8_t {
    kUnset,
    kPending,
    kSet,
  };

  struct stable_data_t {
    stable_data_t(Node* p, core::action_index_t a);
    stable_data_t(Node* p, core::action_index_t a, const Tensorizor&, const GameState&, const GameOutcome&);
    stable_data_t(const stable_data_t& data, bool prune_parent);

    int num_valid_actions() const { return valid_action_mask.count(); }  // consider saving in member variable

    Node* parent;
    Tensorizor tensorizor;
    GameState state;
    GameOutcome outcome;
    ActionMask valid_action_mask;
    core::action_index_t action;
    core::seat_index_t current_player;
    core::symmetry_index_t sym_index;

  private:
    void aux_init();
  };

  /*
   * We represent the children of a node as a std::array of Node*. The i'th element of the array corresponds to the
   * i'th set-bit of the valid_actions mask. The children are lazily expanded.
   *
   * The array representation can be wasteful if the number of children is small, but it's simple and good enough for
   * now. A less wasteful approach might use a vector or linked-list, but this would require more complicated memory
   * management.
   *
   * TODO: add begin()/end() methods to allow for cleaner iteration over child Node* pointers.
   */
  struct children_data_t {
    using array_t = std::array<Node*, kMaxNumLocalActions>;

    children_data_t() : array_(), num_children_(0) {}
    Node* operator[](child_index_t c) const { return array_[c]; }
    void set(child_index_t c, Node* child) { array_[c] = child; }
    void clear(child_index_t c) { array_[c] = nullptr; }
    int num_children() const { return num_children_; }

  private:
    array_t array_;
    int num_children_;
  };

  struct evaluation_data_t {
    evaluation_data_t(const ActionMask& valid_actions);

    NNEvaluation::asptr ptr;
    LocalPolicyArray local_policy_prob_distr;
    evaluation_state_t state = kUnset;
    ActionMask fully_analyzed_actions;  // means that every leaf descendent is a terminal game state
  };

  /*
   * Thread-safety policy: mutex on writes, not on reads. On reads, we simply do a copy of the entire struct, in
   * order to simplify the reasoning about race-conditions.
   *
   * Note that for the non-primitive members, the writes are not guaranteed to be atomic. A non-mutex-protected-read
   * may encounter partially-updated arrays when reading such members. Furthermore, there are no guarantees in this
   * implementation of the order of member-updates when writing, meaning that non-mutex-protected-reads might
   * encounter states where some of the members have been updated while other have not.
   *
   * Despite the above caveats, we can still read without a mutex, since all usages are ok with arbitrarily-partially
   * written data.
   */
  struct stats_t {
    stats_t();
    void zero_out();
    void remove(const ValueArray& rm_sum, int rm_count);

    ValueArray value_avg;
    int count = 0;
    int virtual_count = 0;  // only used for debugging
    player_bitset_t forcibly_winning;  // used for eliminations
    player_bitset_t forcibly_losing;  // used for eliminations
  };

  Node(Node* parent, core::action_index_t action);
  Node(const Tensorizor&, const GameState&, const GameOutcome&);
  Node(const Node& node, bool prune_parent=false);

  std::string genealogy_str() const;  // slow, for debugging
  void debug_dump() const;

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
  void release(Node* protected_child= nullptr);

  /*
   * Set child->parent = this for all children of this.
   *
   * This is the only reason that stable_data_ is not const.
   */
  void adopt_children();

  std::condition_variable& cv_evaluate_and_expand() { return cv_evaluate_and_expand_; }
  std::mutex& evaluation_data_mutex() const { return evaluation_data_mutex_; }
  std::mutex& stats_mutex() const { return stats_mutex_; }

  PolicyTensor get_counts() const;
  void backprop(const ValueArray& value);
  void backprop_with_virtual_undo(const ValueArray& value);
  void virtual_backprop();
  void eliminate(int thread_id, player_bitset_t& forcibly_winning, player_bitset_t& forcibly_losing,
                 ValueArray& accumulated_value, int& accumulated_count);
  void compute_forced_lines(player_bitset_t& forcibly_winning, player_bitset_t& forcibly_losing) const;

  bool forcibly_winning(const stats_t& stats) const { return stats.forcibly_winning[stable_data_.current_player]; }
  bool forcibly_losing(const stats_t& stats) const { return stats.forcibly_losing[stable_data_.current_player]; }
  bool eliminated(const stats_t& stats) const { return forcibly_winning(stats) || forcibly_losing(stats); }

  bool forcibly_winning() const { return forcibly_winning(stats_); }
  bool forcibly_losing() const { return forcibly_losing(stats_); }
  bool eliminated() const { return eliminated(stats_); }

  ValueArray make_virtual_loss() const;
  void mark_as_fully_analyzed();

  const stable_data_t& stable_data() const { return stable_data_; }
  core::action_index_t action() const { return stable_data_.action; }
  Node* parent() const { return stable_data_.parent; }
  bool is_root() const { return !stable_data_.parent; }

  bool has_children() const { return children_data_.num_children(); }
  int num_children() const { return children_data_.num_children(); }
  Node* get_child(child_index_t c) const { return children_data_[c]; }
  void clear_child(child_index_t c) { children_data_.clear(c); }
  Node* init_child(child_index_t c);
  Node* lookup_child_by_action(core::action_index_t action) const;

  const stats_t& stats() const { return stats_; }

  const evaluation_data_t& evaluation_data() const { return evaluation_data_; }
  evaluation_data_t& evaluation_data() { return evaluation_data_; }

private:
  std::condition_variable cv_evaluate_and_expand_;
  mutable std::mutex evaluation_data_mutex_;
  mutable std::mutex children_mutex_;
  mutable std::mutex stats_mutex_;
  stable_data_t stable_data_;  // effectively const
  children_data_t children_data_;
  evaluation_data_t evaluation_data_;
  stats_t stats_;
};

}  // namespace mcts

#include <mcts/inl/Node.inl>
