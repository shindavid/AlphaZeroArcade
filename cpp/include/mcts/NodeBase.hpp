#pragma once

#include "core/concepts/Game.hpp"
#include "mcts/Edge.hpp"
#include "mcts/TypeDefs.hpp"
#include "util/AllocPool.hpp"

namespace mcts {

/*
 * StateData<Game, false> is an empty class that does nothing.
 *
 * StateData<Game, true> is a class that stores a game state.
 *
 * NodeBase<Game>::StableData inherits from StateData<Game, B>, with B set to true only if the
 * macro STORE_STATES is enabled.
 *
 * This allows for us to store the game state in the node object, which is useful for debugging and
 * analysis.
 */
template <core::concepts::Game Game, bool EnableStorage>
struct StateData {
  using State = Game::State;

  StateData(const State&) {}
  const State* get_state() const { return nullptr; }
};

template <core::concepts::Game Game>
struct StateData<Game, true> {
  using State = Game::State;

  StateData(const State& s) : state(s) {}
  const State* get_state() const { return &state; }

  State state;
};

// mcts::NodeBase<Game> is a base class of mcts::Node<Game>.
//
// It consists of data and methods that are shared across all different search-frameworks
// (e.g., MCTS and Bayesian-MCTS)

template <core::concepts::Game Game, typename Derived>
class NodeBase {
 public:
  using NodeDerived = Derived;
  using StateHistory = Game::StateHistory;
  using ActionMask = Game::Types::ActionMask;
  using ValueTensor = Game::Types::ValueTensor;

  static constexpr bool kStoreStates =
    IS_DEFINED(STORE_STATES) || Game::MctsConfiguration::kStoreStates;
  using StateData = mcts::StateData<Game, kStoreStates>;

  // We make the StateData a base-class of StableData because (1) the state is stable, and
  // (2) if STORE_STATES is not enabled, we get an empty base-class optimization.
  struct StableData : public StateData {
    StableData(const StateHistory&, core::seat_index_t active_seat);   // for non-terminal nodes
    StableData(const StateHistory&, const ValueTensor& game_outcome);  // for terminal nodes

    ValueTensor VT;
    ActionMask valid_action_mask;
    int num_valid_actions;
    core::action_mode_t action_mode;

    // active_seat is usually the current player, who is about to make a move
    // if this is a chance node, active_seat is the player who just made a move
    core::seat_index_t active_seat;

    bool terminal;
    bool VT_valid;
    bool is_chance_node;
  };

  class LookupTable {
   public:
    using MCTSKey = Game::InputTensorizor::MCTSKey;
    using NodeBase = mcts::NodeBase<Game, Derived>;

    class Defragmenter {
     public:
      Defragmenter(LookupTable* table);
      void scan(node_pool_index_t);
      void prepare();
      void remap(node_pool_index_t&);
      void defrag();

     private:
      using bitset_t = boost::dynamic_bitset<>;
      using index_vec_t = std::vector<util::pool_index_t>;

      void remap_helper(node_pool_index_t, bitset_t&);
      static void init_remapping(index_vec_t&, bitset_t&);

      LookupTable* table_;
      bitset_t node_bitset_;
      bitset_t edge_bitset_;

      index_vec_t node_index_remappings_;
      index_vec_t edge_index_remappings_;
    };

    LookupTable(mutex_vec_sptr_t mutex_pool);
    LookupTable(const LookupTable&) = delete;
    LookupTable& operator=(const LookupTable&) = delete;

    void clear();

    void defragment(node_pool_index_t& root_index);

    // Inserts a mapping from k to v.
    //
    // If overwrite is true, the mapping is inserted regardless of whether k is already in the map.
    // Else, the mapping is only inserted if k is not already in the map.
    //
    // Returns the value that k maps to after the operation.
    node_pool_index_t insert_node(const MCTSKey& k, node_pool_index_t v, bool overwrite);

    // Returns the value that k maps to, or -1 if k is not in the map.
    node_pool_index_t lookup_node(const MCTSKey&) const;

    node_pool_index_t alloc_node() { return node_pool_.alloc(1); }
    edge_pool_index_t alloc_edges(int k) { return edge_pool_.alloc(k); }
    const NodeDerived* get_node(node_pool_index_t index) const { return &node_pool_[index]; }
    NodeDerived* get_node(node_pool_index_t index) { return &node_pool_[index]; }
    const Edge* get_edge(edge_pool_index_t index) const { return &edge_pool_[index]; }
    Edge* get_edge(edge_pool_index_t index) { return &edge_pool_[index]; }

    using map_t = std::unordered_map<MCTSKey, node_pool_index_t>;
    const map_t* map() const { return &map_; }

    int get_random_mutex_id() const;
    mit::mutex& get_mutex(int mutex_id);

   private:
    friend class Defragmenter;
    map_t map_;
    util::AllocPool<Edge> edge_pool_;
    util::AllocPool<NodeDerived> node_pool_;
    mutex_vec_sptr_t mutex_pool_;
    const int mutex_pool_size_;
    mutable mit::mutex map_mutex_;
  };

  NodeBase(LookupTable*, const StateHistory&, core::seat_index_t active_seat);   // for non-terminal
  NodeBase(LookupTable*, const StateHistory&, const ValueTensor& game_outcome);  // for terminal

  const StableData& stable_data() const { return stable_data_; }

  mit::mutex& mutex() const { return this->lookup_table_->get_mutex(mutex_id_); }

  bool is_terminal() const { return stable_data_.terminal; }
  core::action_mode_t action_mode() const { return stable_data_.action_mode; }
  void initialize_edges();

  NodeDerived* get_child(const Edge* edge) const;
  node_pool_index_t lookup_child_by_action(core::action_t action) const;
  void update_child_expand_count(int n = 1);
  bool all_children_edges_initialized() const;
  bool edges_initialized() const { return first_edge_index_ != -1; }
  Edge* get_edge(int i) const;
  edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(edge_pool_index_t e) { first_edge_index_ = e; }
  bool trivial() const { return trivial_; }

 protected:
  StableData stable_data_;
  LookupTable* lookup_table_;

  // Each Node has an int mutex_id_, rather than an actual mutex. This is for 2 reasons:
  //
  // 1. Allows multiple Node's to share the same mutex
  // 2. Allows for the Node object to copied and moved around (which is needed for defragmentation)
  int mutex_id_;

  edge_pool_index_t first_edge_index_ = -1;
  int child_expand_count_ = 0;

  bool trivial_ = false;  // set to true if all actions discovered to be symmetrically equivalent
};

}  // namespace mcts

#include "inline/mcts/LookupTable.inl"
#include "inline/mcts/NodeBase.inl"
