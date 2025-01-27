#pragma once

#include <condition_variable>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>

#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/ManagerParams.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/AllocPool.hpp>

namespace mcts {

/*
 * StateData<Game, false> is an empty class that does nothing.
 *
 * StateData<Game, true> is a class that stores a game state.
 *
 * Node::StableData inherits from StateData, with the bool argument set to true only if the
 * macro STORE_STATES is enabled.
 *
 * This allows for us to store the game state in the Node object, which is useful for debugging and
 * analysis.
 */
template <core::concepts::Game Game, bool EnableStorage>
struct StateData {
  using State = typename Game::State;

  StateData(const State&) {}
  const State* get_state() const { return nullptr; }
};

template <core::concepts::Game Game>
struct StateData<Game, true> {
  using State = typename Game::State;

  StateData(const State& s) : state(s) {}
  const State* get_state() const { return &state; }

  State state;
};

/*
 * A Node consists of n=3 main data member:
 *
 * StableData: write-once data that is fixed for the lifetime of the node
 * Stats: values that get updated throughout MCTS via backpropagation
 * Edge[]: edges to children nodes, needed for tree traversal
 *
 * The last one, Edge[], is represented by a single index into a pool of edges.
 *
 * During MCTS, multiple search threads will try to read and write these values. Thread-safety is
 * achieved in a high-performance manner through mutexes and condition variables.
 */
template <core::concepts::Game Game>
class Node {
 public:
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;
  static constexpr int kNumPlayers = Game::Constants::kNumPlayers;

  using ManagerParams = mcts::ManagerParams<Game>;
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using MCTSKey = Game::InputTensorizor::MCTSKey;
  using ActionMask = Game::Types::ActionMask;
  using LocalPolicyArray = Game::Types::LocalPolicyArray;
  using LocalActionValueArray = Game::Types::LocalActionValueArray;
  using ValueArray = Game::Types::ValueArray;
  using ValueTensor = Game::Types::ValueTensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;
  using SearchResults = Game::Types::SearchResults;
  using player_bitset_t = Game::Types::player_bitset_t;
  using node_pool_index_t = util::pool_index_t;
  using edge_pool_index_t = util::pool_index_t;

  enum expansion_state_t : int8_t {
    kNotExpanded,
    kMidExpansion,
    kPreExpanded,  // used when evaluating all children when computing AV-targets
    kExpanded
  };

  static constexpr bool kStoreStates =
      IS_MACRO_ENABLED(STORE_STATES) || Game::MctsConfiguration::kStoreStates;
  using StateData = mcts::StateData<Game, kStoreStates>;

  // We make the StateData a base-class of StableData because (1) the state is stable, and
  // (2) if STORE_STATES is not enabled, we get an empty base-class optimization.
  struct StableData : public StateData {
    StableData(const StateHistory&, core::seat_index_t active_seat);  // for non-terminal nodes
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

  // Generally, we acquire this->mutex() when reading or writing to this->stats_. There are some
  // exceptions on reads, when we read a single atomically-writable member of stats_.
  struct Stats {
    int total_count() const { return RN + VN; }
    void init_q(const ValueArray&, bool pure);
    void update_provable_bits(const player_bitset_t& all_actions_provably_winning,
                              const player_bitset_t& all_actions_provably_losing,
                              int num_expanded_children, bool cp_has_winning_move,
                              const StableData&);

    ValueArray Q;     // excludes virtual loss
    ValueArray Q_sq;  // excludes virtual loss
    int RN = 0;       // real count
    int VN = 0;       // virtual count

    // TODO: generalize these fields to utility lower/upper bounds
    player_bitset_t provably_winning;
    player_bitset_t provably_losing;
  };

  /*
   * An Edge corresponds to an action that can be taken from this node.
   */
  struct Edge {
    node_pool_index_t child_index = -1;
    core::action_t action = -1;
    int E = 0;  // real or virtual count
    float base_prob = 0;  // used for both raw policy prior and chance node probability
    // equal to base_prob, with possible adjustments from Dirichlet-noise and softmax-temperature
    float adjusted_base_prob = 0;
    float child_V_estimate = 0;  // network estimate of child-value for current-player
    group::element_t sym = -1;
    expansion_state_t state = kNotExpanded;
  };

  class LookupTable {
   public:
    static constexpr int kDefaultMutexPoolSize = 256;

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

    LookupTable(bool multithreaded_mode);
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
    const Node* get_node(node_pool_index_t index) const { return &node_pool_[index]; }
    Node* get_node(node_pool_index_t index) { return &node_pool_[index]; }
    const Edge* get_edge(edge_pool_index_t index) const { return &edge_pool_[index]; }
    Edge* get_edge(edge_pool_index_t index) { return &edge_pool_[index]; }

    using map_t = std::unordered_map<MCTSKey, node_pool_index_t>;
    const map_t* map() const { return &map_; }

    int get_random_mutex_id() const;
    std::mutex& get_mutex(int mutex_id) { return mutex_pool_[mutex_id]; }
    std::condition_variable& get_cv(int mutex_id) { return cv_pool_[mutex_id]; }

   private:
    friend class Defragmenter;
    map_t map_;
    util::AllocPool<Edge> edge_pool_;
    util::AllocPool<Node> node_pool_;
    std::vector<std::mutex> mutex_pool_;
    std::vector<std::condition_variable> cv_pool_;
    mutable std::mutex map_mutex_;
  };

  Node(LookupTable*, const StateHistory&, core::seat_index_t active_seat);  // for non-terminal
  Node(LookupTable*, const StateHistory&, const ValueTensor& game_outcome);  // for terminal

  void write_results(const ManagerParams& params, group::element_t inv_sym,
                     SearchResults& results) const;

  template <typename MutexProtectedFunc>
  void update_stats(MutexProtectedFunc);

  node_pool_index_t lookup_child_by_action(core::action_t action) const;

  const StableData& stable_data() const { return stable_data_; }

  // stats() returns a reference to the stats object, WITHOUT acquiring the mutex. In order to use
  // this function properly, the caller must ensure that one of the following is true:
  //
  // 1. The context is single-threaded,
  //
  // or,
  //
  // 2. The usage of the stats reference falls within the scope of the node's mutex,
  //
  // or,
  //
  // 3. The caller is ok with the possibility of a race-condition with a writer.
  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }

  // Acquires the mutex and returns a copy of the stats object.
  Stats stats_safe() const;

  bool is_terminal() const { return stable_data_.terminal; }
  core::action_mode_t action_mode() const { return stable_data_.action_mode; }

  std::mutex& mutex() const { return lookup_table_->get_mutex(mutex_id_); }
  std::condition_variable& cv() const { return lookup_table_->get_cv(mutex_id_); }

  void initialize_edges();

  template<typename PolicyTransformFunc>
  void load_eval(NNEvaluation* eval, PolicyTransformFunc);

  bool all_children_edges_initialized() const;
  bool edges_initialized() const { return first_edge_index_ != -1; }
  Edge* get_edge(int i) const;
  edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(edge_pool_index_t e) { first_edge_index_ = e; }
  Node* get_child(const Edge* edge) const;
  void update_child_expand_count(int n=1);
  bool trivial() const { return trivial_; }

  // NO-OP in release builds, checks various invariants in debug builds
  void validate_state() const;

 private:
  StableData stable_data_;
  LookupTable* lookup_table_;
  Stats stats_;

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

#include <inline/mcts/Node.inl>
