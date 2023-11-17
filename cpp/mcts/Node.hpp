#pragma once

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>

#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/TypeDefs.hpp>

namespace mcts {

/*
 * A Node consists of n=3 main groups of non-const member variables:
 *
 * children_data_: edges to children nodes, needed for tree traversal
 * evaluation_data_: policy/value vectors that come from neural net evaluation
 * stats_: values that get updated throughout MCTS via backpropagation
 *
 * During MCTS, multiple search threads will try to read and write these values. Thread-safety is
 * achieved in a high-performance manner through mutexes and condition variables.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class Node {
 public:
  using sptr = std::shared_ptr<Node>;

  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using GameStateTypes = core::GameStateTypes<GameState>;

  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;
  static constexpr int kNumGlobalActions = GameStateTypes::kNumGlobalActions;
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kEdgeDataChunkSize = std::min(8, kMaxNumLocalActions);

  using Action = typename GameStateTypes::Action;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;
  using dtype = typename GameStateTypes::dtype;

  using player_bitset_t = std::bitset<kNumPlayers>;

  enum evaluation_state_t : int8_t {
    kUnset,
    kSet,
  };

  struct stable_data_t {
    stable_data_t(const Tensorizor&, const GameState&, const GameOutcome&, const ManagerParams*);

    Tensorizor tensorizor;
    GameState state;
    GameOutcome outcome;
    ActionMask valid_action_mask;
    int num_valid_actions;
    core::seat_index_t current_player;
    core::symmetry_index_t sym_index;
  };

  /*
   * Thread-safety policy: mutex on writes, not on reads.
   *
   * Note that for the non-primitive members (i.e., the members of type ValueArray), the writes are
   * not guaranteed to be atomic. A non-mutex-protected-read may encounter partially-updated arrays
   * when reading such members. Furthermore, there are no guarantees in this implementation of the
   * order of member-updates when writing, meaning that non-mutex-protected-reads might encounter
   * states where some of the members have been updated while other have not.
   *
   * Despite the above caveats, we can still read without a mutex, since all usages are ok with
   * arbitrarily-partially written data.
   *
   * TODO: There are a number of ways to shrink the memory footprint of this object:
   *
   * 1. I don't think there's a need to have both real_avg and virtualized_avg, as the virtualized
   *    avg can be quickly computed as needed from the real avg and the counts.
   * 2. In zero-sum games, ValueArray can be specialized to only store (n-1) values, and packed.
   * 3. The provably_* bitsets can be changed to use a specialized bitset type that uses fewer than
   *    8 bytes.
   *
   * With the above changes, this object can be shrunk to 16 bytes in the two-player zero-sum case.
   * This shrinkage became slightly more relevant with the introduction of Deterministic
   * Multithreaded MCTS, which maintains two stats_t objects per node, one for prefetched data and
   * one for confirmed data.
   */
  struct stats_t {
    stats_t();
    int total_count() const { return real_count + virtual_count; }
    void virtual_increment() { virtual_count++; }
    void real_increment() { real_count++; }
    void increment_transfer() {
      real_count++;
      virtual_count--;
    }
    void deduce_certain_outcomes(const ValueArray& value) {
      for (int p = 0; p < kNumPlayers; ++p) {
        provably_winning[p] = value(p) == 1;
        provably_losing[p] = value(p) == 0;
      }
    }
    void set_eval_with_virtual_undo(const ValueArray& value) {
      increment_transfer();
    }

    ValueArray real_avg;         // excludes virtual loss
    ValueArray virtualized_avg;  // includes virtual loss

    // TODO: generalize these fields to utility lower/upper bounds
    player_bitset_t provably_winning;
    player_bitset_t provably_losing;

    int real_count = 0;
    int virtual_count = 0;
  };

  /*
   * An edge_t corresponds to an action that can be taken from this node. It is instantiated
   * only when the action is expanded. It stores both a (smart) pointer to the child node and an
   * edge count. The edge count is used to support MCTS mechanics.
   *
   * When instantiating an edge_t, we assign the members in the order child_, action_,
   * action_index_. The instantiated() check looks at the last of these (action_index_). This
   * discipline ensures that lock-free usages work properly. Write ordering is enforced via the
   * volatile keyword.
   */
  struct edge_t {
    edge_t();
    edge_t* instantiate(const Action& a, core::action_index_t i, sptr c);
    bool instantiated() const { return action_index_ >= 0; }
    Action action() const { return const_cast<Action&>(action_); }
    core::action_index_t action_index() const { return action_index_; }
    sptr child() const { return const_cast<sptr&>(child_); }
    void increment_count(TreeTraversalMode mode) { count_[mode]++; }
    int count(TreeTraversalMode mode) const { return count_[mode].load(); }
    void reset_count() { count_[kPrefetchMode].store(count(kSearchMode)); }

     private:
      volatile sptr child_;
      volatile Action action_;
      volatile core::action_index_t action_index_ = -1;
      std::atomic<int> count_[kNumTreeTraversalModes] = {0, 0};  // real only
  };

  /*
   * A chunk of edge_t's, together with a pointer to the next chunk. This chunking results in
   * more efficient memory access patterns. In particular, if a node is not expanded to more than
   * kEdgeDataChunkSize children, then we avoid dynamic memory allocation.
   */
  struct edge_chunk_t {
    ~edge_chunk_t() { delete next; }
    edge_t* find(core::action_index_t i);
    edge_t* insert(const Action& a, core::action_index_t i, sptr child);

    edge_t data[kEdgeDataChunkSize];
    edge_chunk_t* next = nullptr;
  };

  /*
   * children_data_t maintains a logical map of action -> edge_t. It is implemented as a
   * chunked linked list. Compared to a more natural std::map representation, lookups are
   * theoretically slower (O(N) vs O(log(N))). However, the linked list representation offers
   * notable advantages:
   *
   * - Allows us to do lock-free reads. This is because data addresses are stable in linked lists,
   *   unlike in std maps/vectors. Note that reads are much more frequent than writes in MCTS.
   *
   * - The mechanics of MCTS are such that the most frequently visited children are likely to be at
   *   the front of the linked list, which may make the worst-time O(N) be more like an
   *   average-case O(1) in practice.
   *
   * When appending to children_data_t, we need to grab children_mutex_.
   *
   * TODO: use a custom allocator for the linked list.
   */
  struct children_data_t {
    template <bool is_const>
    struct iterator_base_t {
      using chunk_t = std::conditional_t<is_const, const edge_chunk_t, edge_chunk_t>;

      iterator_base_t(chunk_t* chunk = nullptr, int index = 0);
      bool operator==(const iterator_base_t& other) const = default;

     protected:
      void increment();
      void nullify_if_at_end();

      chunk_t* chunk;
      int index;
    };

    struct iterator : public iterator_base_t<false> {
      using base_t = iterator_base_t<false>;

      using iterator_category = std::forward_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = edge_t;
      using pointer = value_type*;
      using reference = value_type&;

      using base_t::base_t;
      iterator& operator++() {
        this->increment();
        return *this;
      }
      iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }
      bool operator==(const iterator& other) const = default;
      bool operator!=(const iterator& other) const = default;
      edge_t& operator*() const { return this->chunk->data[this->index]; }
      edge_t* operator->() const { return &this->chunk->data[this->index]; }
    };
    static_assert(std::forward_iterator<iterator>);

    struct const_iterator : public iterator_base_t<true> {
      using base_t = iterator_base_t<true>;

      using iterator_category = std::forward_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = edge_t;
      using pointer = value_type*;
      using reference = value_type&;

      using base_t::base_t;
      const_iterator& operator++() {
        this->increment();
        return *this;
      }
      const_iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }
      bool operator==(const const_iterator& other) const = default;
      bool operator!=(const const_iterator& other) const = default;
      const edge_t& operator*() const { return this->chunk->data[this->index]; }
      const edge_t* operator->() const { return &this->chunk->data[this->index]; }
    };
    static_assert(std::forward_iterator<const_iterator>);

    /*
     * Traverses the chunked linked list and attempts to find an edge_t corresponding to the
     * given action. If it finds it, then it returns a pointer to the edge_t. Otherwise, it
     * returns nullptr.
     *
     * The expected usage is to call this first without grabbing children_mutex_, to optimize for
     * the more common case where the action has already been expanded. If the action has not been
     * expanded, then we grab children_mutex_ and call insert(). Note that there is a race-condition
     * possible; insert() deals with this possibility appropriately.
     */
    edge_t* find(core::action_index_t i) { return first_chunk_.find(i); }

    /*
     * Inserts a new edge_t into the chunked linked list with the given action/Node, and
     * returns a pointer to it.
     *
     * It is possible that an edge_t already exists for this action due to a race condition.
     * In this case, returns a pointer to the existing entry.
     */
    edge_t* insert(const Action& a, core::action_index_t i, sptr child) {
      return first_chunk_.insert(a, i, child);
    }

    iterator begin() { return iterator(&first_chunk_, 0); }
    iterator end() { return iterator(nullptr, 0); }
    const_iterator begin() const { return const_iterator(&first_chunk_, 0); }
    const_iterator end() const { return const_iterator(nullptr, 0); }

   private:
    edge_chunk_t first_chunk_;
  };

  struct evaluation_data_t {
    NNEvaluation::asptr ptr;
    LocalPolicyArray local_policy_prob_distr;

    /*
     * For non-terminal nodes, value is a copy of ptr->value_prob_distr_, and represents a
     * neural-net evaluation of the node.
     *
     * For terminal nodes, value is a copy of stable_data_.outcome, and represents an outcome as
     * determined by the rules of the game.
     *
     * Either way, value is a redundant copied value. We don't need this field, and could instead
     * have a getter that branches based on whether the node is terminal or not. Having a copied
     * value is likely a tiny efficiency win by virtue of skipping branches and pointer
     * redirections.
     */
    ValueArray value;
    evaluation_state_t state = kUnset;
  };

  Node(const Tensorizor&, const GameState&, const GameOutcome&, const ManagerParams*);

  void debug_dump() const;

  std::mutex& evaluation_data_mutex() const { return evaluation_data_mutex_; }
  std::mutex& stats_mutex() const { return stats_mutex_; }

  PolicyTensor get_counts(const ManagerParams& params) const;
  ValueArray make_virtual_loss() const;
  template <typename UpdateT>
  void update_stats(const UpdateT& update_instruction, TreeTraversalMode mode);
  sptr lookup_child_by_action(const Action& action) const;

  /*
   * Copies search-mode stats to prefetch-mode stats. Recursively does this for all children.
   *
   * Assumes that all other threads besides the current one have stopped working on this tree.
   */
  void reset_prefetch_stats();

  const stable_data_t& stable_data() const { return stable_data_; }
  const children_data_t& children_data() const { return children_data_; }
  children_data_t& children_data() { return children_data_; }
  std::mutex& children_mutex() { return children_mutex_; }
  const stats_t& stats(TreeTraversalMode mode) const { return stats_[mode]; }
  stats_t& stats(TreeTraversalMode mode) { return stats_[mode]; }
  const evaluation_data_t& evaluation_data() const { return evaluation_data_; }
  evaluation_data_t& evaluation_data() { return evaluation_data_; }

 private:
  static core::symmetry_index_t make_sym_index(const GameState& state, const ManagerParams& params);
  mutable std::mutex evaluation_data_mutex_;
  mutable std::mutex children_mutex_;
  mutable std::mutex stats_mutex_;  // protects usage of stats_[kPrefetchMode]
  const stable_data_t stable_data_;
  children_data_t children_data_;
  evaluation_data_t evaluation_data_;
  stats_t stats_[kNumTreeTraversalModes];
};

}  // namespace mcts

#include <mcts/inl/Node.inl>
