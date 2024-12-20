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
 * Node::stable_data_t inherits from StateData, with the bool argument set to true only if the
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
 * stable_data_t: write-once data that is fixed for the lifetime of the node
 * stats_t: values that get updated throughout MCTS via backpropagation
 * edge_t[]: edges to children nodes, needed for tree traversal
 *
 * The last one, edge_t[], is represented by a single index into a pool of edges.
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
  using player_bitset_t = std::bitset<kNumPlayers>;
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

  // We make the StateData a base-class of stable_data_t because (1) the state is stable, and
  // (2) if STORE_STATES is not enabled, we get an empty base-class optimization.
  struct stable_data_t : public StateData {
    stable_data_t(const StateHistory&);  // for non-terminal nodes
    stable_data_t(const StateHistory&, const ValueTensor& game_outcome);  // for terminal nodes

    ValueTensor VT;
    ActionMask valid_action_mask;
    int num_valid_actions;
    core::action_mode_t action_mode;
    core::seat_index_t current_player;
    bool terminal;
    bool VT_valid;
    bool prior_prob_known;
    PolicyTensor prior_prob;
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
   */
  struct stats_t {
    int total_count() const { return RN + VN; }
    void init_q(const ValueArray&, bool pure);

    ValueArray Q;     // excludes virtual loss
    ValueArray Q_sq;  // excludes virtual loss
    int RN = 0;       // real count
    int VN = 0;       // virtual count

    // TODO: generalize these fields to utility lower/upper bounds
    player_bitset_t provably_winning;
    player_bitset_t provably_losing;
  };

  /*
   * An edge_t corresponds to an action that can be taken from this node.
   */
  struct edge_t {
    node_pool_index_t child_index = -1;
    core::action_t action = -1;
    int E = 0;  // real or virtual count
    float raw_policy_prior = 0;
    float adjusted_policy_prior = 0;
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
    void insert_node(const MCTSKey&, node_pool_index_t);
    node_pool_index_t lookup_node(const MCTSKey&) const;

    node_pool_index_t alloc_node() { return node_pool_.alloc(1); }
    edge_pool_index_t alloc_edges(int k) { return edge_pool_.alloc(k); }
    const Node* get_node(node_pool_index_t index) const { return &node_pool_[index]; }
    Node* get_node(node_pool_index_t index) { return &node_pool_[index]; }
    const edge_t* get_edge(edge_pool_index_t index) const { return &edge_pool_[index]; }
    edge_t* get_edge(edge_pool_index_t index) { return &edge_pool_[index]; }

    using map_t = std::unordered_map<MCTSKey, node_pool_index_t>;
    const map_t* map() const { return &map_; }

    int get_random_mutex_id() const;
    std::mutex& get_mutex(int mutex_id) { return mutex_pool_[mutex_id]; }
    std::condition_variable& get_cv(int mutex_id) { return cv_pool_[mutex_id]; }

   private:
    friend class Defragmenter;
    map_t map_;
    util::AllocPool<edge_t> edge_pool_;
    util::AllocPool<Node> node_pool_;
    std::vector<std::mutex> mutex_pool_;
    std::vector<std::condition_variable> cv_pool_;
    mutable std::mutex map_mutex_;
  };

  Node(LookupTable*, const StateHistory&);  // for non-terminal nodes
  Node(LookupTable*, const StateHistory&, const ValueTensor& game_outcome);  // for terminal nodes

  void write_results(const ManagerParams& params, group::element_t inv_sym,
                     SearchResults& results) const;

  template <typename MutexProtectedFunc>
  void update_stats(MutexProtectedFunc);

  node_pool_index_t lookup_child_by_action(core::action_t action) const;

  const stable_data_t& stable_data() const { return stable_data_; }
  const stats_t& stats() const { return stats_; }
  stats_t& stats() { return stats_; }
  bool is_terminal() const { return stable_data_.terminal; }
  core::action_mode_t action_mode() const { return stable_data_.action_mode; }

  std::mutex& mutex() const { return lookup_table_->get_mutex(mutex_id_); }
  std::condition_variable& cv() { return lookup_table_->get_cv(mutex_id_); }

  void initialize_edges();

  template<typename PolicyTransformFunc>
  void load_eval(NNEvaluation* eval, PolicyTransformFunc);

  bool all_children_edges_initialized() const;
  bool edges_initialized() const { return first_edge_index_ != -1; }
  edge_t* get_edge(int i) const;
  edge_pool_index_t get_first_edge_index() const { return first_edge_index_; }
  void set_first_edge_index(edge_pool_index_t e) { first_edge_index_ = e; }
  Node* get_child(const edge_t* edge) const;
  void update_child_expand_count(int n=1);
  bool trivial() const { return trivial_; }

  // NO-OP in release builds, checks various invariants in debug builds
  void validate_state() const;

 private:
  stable_data_t stable_data_;
  LookupTable* lookup_table_;
  stats_t stats_;
  int mutex_id_;
  edge_pool_index_t first_edge_index_ = -1;
  int child_expand_count_ = 0;
  bool trivial_ = false;  // set to true if all actions discovered to be symmetrically equivalent
};

}  // namespace mcts

#include <inline/mcts/Node.inl>
