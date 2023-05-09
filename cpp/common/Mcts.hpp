#pragma once

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <Eigen/Core>
#include <EigenRand/EigenRand>

#include <common/AbstractSymmetryTransform.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/MctsResults.hpp>
#include <common/NeuralNet.hpp>
#include <common/TensorizorConcept.hpp>
#include <util/AtomicSharedPtr.hpp>
#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenTorch.hpp>
#include <util/LRUCache.hpp>
#include <util/Math.hpp>
#include <util/Profiler.hpp>

namespace common {

/*
 * TODO: move the various inner-classes of Mcts into separate files as standalone-classes. Proposed class-renaming:
 *
 * common::Mcts<GameState, Tensorizor> -> mcts::Tree<GameState, Tensorizor>
 * common::Mcts<GameState, Tensorizor>::Params -> mcts::Params<GameState, Tensorizor>
 * common::Mcts<GameState, Tensorizor>::SearchThread -> mcts::SearchThread<GameState, Tensorizor>
 * common::Mcts<GameState, Tensorizor>::Node -> mcts::Node<GameState, Tensorizor>
 *
 * etc.
 *
 * TODO: use CRTP for slightly more elegant inheritance mechanics.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class Mcts {
private:
  class SearchThread;

public:
  using TensorizorTypes = common::TensorizorTypes<Tensorizor>;
  using GameStateTypes = common::GameStateTypes<GameState>;

  static constexpr bool kEnableProfiling = IS_MACRO_ASSIGNED_TO_1(PROFILE_MCTS);
  static constexpr bool kEnableVerboseProfiling = IS_MACRO_ASSIGNED_TO_1(PROFILE_MCTS_VERBOSE);
  static constexpr bool kEnableThreadingDebug = IS_MACRO_ASSIGNED_TO_1(MCTS_THREADING_DEBUG);
  static constexpr bool kDeterministic = IS_MACRO_ASSIGNED_TO_1(DETERMINISTIC_MCTS);

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameStateTypes::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using dtype = typename GameStateTypes::dtype;
  using child_index_t = int;

  using MctsResults = common::MctsResults<GameState>;
  using SymmetryTransform = AbstractSymmetryTransform<GameState, Tensorizor>;
  using ValueProbDistr = typename GameStateTypes::ValueProbDistr;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ActionMask = typename GameStateTypes::ActionMask;
  using LocalPolicyLogitDistr = typename GameStateTypes::LocalPolicyLogitDistr;
  using LocalPolicyProbDistr = typename GameStateTypes::LocalPolicyProbDistr;
  using PolicyEigenTensor = typename GameStateTypes::PolicyEigenTensor;
  using ValueEigenTensor = typename GameStateTypes::ValueEigenTensor;

  using InputShape = typename TensorizorTypes::InputShape;
  using PolicyShape = typename GameStateTypes::PolicyShape;

  using FullInputTensor = typename TensorizorTypes::DynamicInputTensor;
  using FullValueTensor = typename GameStateTypes::DynamicValueTensor;
  using FullPolicyTensor = typename GameStateTypes::DynamicPolicyTensor;
  using ValueArray = typename GameStateTypes::ValueArray;
  using PolicyArray = typename GameStateTypes::PolicyArray;

  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

  struct ValueArrayExtrema {
    ValueArray min;
    ValueArray max;
  };

  enum DefaultParamsType {
    kCompetitive,
    kTraining
  };

  /*
   * Params pertains to a single Mcts instance.
   *
   * By contrast, SearchParams pertains to each individual search() call.
   */
  struct Params {
    Params(DefaultParamsType);

    auto make_options_description();
    bool operator==(const Params& other) const = default;

    std::string model_filename;
    int num_search_threads = 8;
    int batch_size_limit = 216;
    bool run_offline = false;
    int offline_tree_size_limit = 4096;
    int64_t nn_eval_timeout_ns = util::us_to_ns(250);
    size_t cache_size = 1048576;

    std::string root_softmax_temperature_str;
    float cPUCT = 1.1;
    float cFPU = 0.2;
    float dirichlet_mult = 0.25;
    float dirichlet_alpha_sum = 0.03 * 361;
    bool disable_eliminations = false;
    bool speculative_evals = false;
    bool forced_playouts = true;
    bool enable_first_play_urgency = true;
    float k_forced = 2.0;
#ifdef PROFILE_MCTS
    std::string profiling_dir;
#endif  // PROFILE_MCTS
  };

  /*
   * SearchParams pertain to a single call to search(). Even given a single Mcts instance, different search() calls can
   * have different SearchParams. For instance, for KataGo, there are "fast" searches and "full" searches, which differ
   * in their tree_size_limit and dirchlet settings.
   *
   * By contrast, Params pertains to a single Mcts instance.
   */
  struct SearchParams {
    static SearchParams make_offline_params(int limit) { return SearchParams{limit, true}; }

    int tree_size_limit = 100;
    bool disable_exploration = false;
  };

private:
  class NNEvaluation {
  public:
    NNEvaluation(const ValueEigenTensor& value, const PolicyEigenTensor& policy, const ActionMask& valid_actions);
    const ValueProbDistr& value_prob_distr() const { return value_prob_distr_; }
    const LocalPolicyProbDistr& local_policy_logit_distr() const { return local_policy_logit_distr_; }

  protected:
    ValueProbDistr value_prob_distr_;
    LocalPolicyLogitDistr local_policy_logit_distr_;
  };
  using NNEvaluation_sptr = std::shared_ptr<NNEvaluation>;
  using NNEvaluation_asptr = util::AtomicSharedPtr<NNEvaluation>;

  /*
   * A Node consists of n=3 main groups of non-const member variables:
   *
   * children_data_: pointers of children nodes, needed for tree traversal
   * evaluation_data_: policy/value vectors that come from neural net evaluation
   * stats_: values that get updated throughout MCTS via backpropagation
   *
   * During MCTS, multiple search threads will try to read and write these values. Thread-safety is achieved in a
   * high-performance manner through a carefully orchestrated combination of mutexes, condition-variables, and
   * lockfree mechanisms.
   */
  class Node {
  public:
    enum evaluation_state_t : int8_t {
      kUnset,
      kPending,
      kSet,
    };

    struct stable_data_t {
      stable_data_t(Node* p, action_index_t a);
      stable_data_t(Node* p, action_index_t a, const Tensorizor&, const GameState&, const GameOutcome&);
      stable_data_t(const stable_data_t& data, bool prune_parent);

      int num_valid_actions() const { return valid_action_mask.count(); }  // consider saving in member variable

      Node* parent;
      action_index_t action;
      Tensorizor tensorizor;
      GameState state;
      GameOutcome outcome;
      ActionMask valid_action_mask;
      seat_index_t current_player;
      symmetry_index_t sym_index;

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

      NNEvaluation_asptr ptr;
      LocalPolicyProbDistr local_policy_prob_distr;
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

      int effective_count() const { return eliminated() ? 0 : count; }
      bool has_certain_outcome() const { return V_floor.sum() > 1 - 1e-6; }  // 1e-6 fudge factor for floating-point error
      bool eliminated() const { return V_floor.maxCoeff() == 1; }  // won/lost positions, not drawn ones
      auto effective_value_avg(seat_index_t s) const { return has_certain_outcome() ? V_floor(s) : value_avg(s); }

      ValueArray value_avg;
      ValueArray V_floor;  // used for eliminations
      int count = 0;
      int virtual_count = 0;  // only used for debugging
    };

    Node(Node* parent, action_index_t action);
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

    PolicyEigenTensor get_effective_counts() const;
    void backprop(const ValueProbDistr& value);
    void backprop_with_virtual_undo(const ValueProbDistr& value);
    void virtual_backprop();
    void perform_eliminations(const ValueProbDistr& outcome);
    ValueArray make_virtual_loss() const;
    void mark_as_fully_analyzed();

    const stable_data_t& stable_data() const { return stable_data_; }
    action_index_t action() const { return stable_data_.action; }
    Node* parent() const { return stable_data_.parent; }
    bool is_root() const { return !stable_data_.parent; }

    bool has_children() const { return children_data_.num_children(); }
    int num_children() const { return children_data_.num_children(); }
    Node* get_child(child_index_t c) const { return children_data_[c]; }
    void clear_child(child_index_t c) { children_data_.clear(c); }
    Node* init_child(child_index_t c);
    Node* lookup_child_by_action(action_index_t action) const;

    const stats_t& stats() const { return stats_; }

    const evaluation_data_t& evaluation_data() const { return evaluation_data_; }
    evaluation_data_t& evaluation_data() { return evaluation_data_; }

  private:
    ValueArrayExtrema get_V_floor_extrema_among_children() const;

    std::condition_variable cv_evaluate_and_expand_;
    mutable std::mutex evaluation_data_mutex_;
    mutable std::mutex children_mutex_;
    mutable std::mutex stats_mutex_;
    stable_data_t stable_data_;  // effectively const
    children_data_t children_data_;
    evaluation_data_t evaluation_data_;
    stats_t stats_;
  };

  class SearchThread {
  public:
    SearchThread(Mcts* mcts, int thread_id);
    ~SearchThread();

    int thread_id() const { return thread_id_; }

    void join();
    void kill();
    void launch(const SearchParams* search_params);
    bool needs_more_visits(Node* root, int tree_size_limit);
    void visit(Node* tree, int depth);

    enum region_t {
      kCheckVisitReady = 0,
      kAcquiringLazilyInitializedDataMutex = 1,
      kLazyInit = 2,
      kBackpropOutcome = 3,
      kPerformEliminations = 4,
      kMisc = 5,
      kCheckingCache = 6,
      kAcquiringBatchMutex = 7,
      kWaitingUntilBatchReservable = 8,
      kTensorizing = 9,
      kIncrementingCommitCount = 10,
      kWaitingForReservationProcessing = 11,
      kVirtualBackprop = 12,
      kConstructingChildren = 13,
      kPUCT = 14,
      kAcquiringStatsMutex = 15,
      kBackpropEvaluation = 16,
      kMarkFullyAnalyzed = 17,
      kEvaluateAndExpand = 18,
      kEvaluateAndExpandUnset = 19,
      kEvaluateAndExpandPending = 20,
      kNumRegions = 21
    };

    using profiler_t = util::Profiler<int(kNumRegions), kEnableVerboseProfiling>;

    void record_for_profiling(region_t region);
    void dump_profiling_stats();
    auto mcts() const { return mcts_; }

#ifdef PROFILE_MCTS
    profiler_t* get_profiler() { return &profiler_; }
    FILE* get_profiling_file() const { return profiling_file_; }
    const char* get_profiler_name() const { return profiler_name_.c_str(); }
    void init_profiling(const char* filename, const char* name) {
      profiling_file_ = fopen(filename, "w");
      profiler_name_ = name;
      profiler_.skip_next_n_dumps(5);
    }
    void close_profiling_file() { fclose(profiling_file_); }

    profiler_t profiler_;
    std::string profiler_name_;
    FILE* profiling_file_ = nullptr;
    int profile_count_ = 0;
#else  // PROFILE_MCTS
    constexpr profiler_t* get_profiler() { return nullptr; }
    static FILE* get_profiling_file() { return nullptr; }
    const char* get_profiler_name() const { return nullptr; }
    void init_profiling(const char* filename, const char* name) {}
    void close_profiling_file() {}
#endif  // PROFILE_MCTS

  private:
    struct evaluate_and_expand_result_t {
      NNEvaluation_sptr evaluation;
      bool backpropagated_virtual_loss;
    };

    void backprop_outcome(Node* tree, const ValueProbDistr& outcome);
    void perform_eliminations(Node* tree, const ValueProbDistr& outcome);
    void mark_as_fully_analyzed(Node* tree);
    evaluate_and_expand_result_t evaluate_and_expand(Node* tree, bool speculative);
    void evaluate_and_expand_unset(
        Node* tree, std::unique_lock<std::mutex>* lock, evaluate_and_expand_result_t* data, bool speculative);
    void evaluate_and_expand_pending(Node* tree, std::unique_lock<std::mutex>* lock);

    /*
     * Used in visit().
     *
     * Applies PUCT criterion to select the best child-index to visit from the given Node.
     *
     * TODO: as we experiment with things like auxiliary NN output heads, dynamic cPUCT values, etc., this method will
     * evolve. It probably makes sense to have the behavior as part of the Tensorizor, since there is coupling with NN
     * architecture (in the form of output heads).
     */
    child_index_t get_best_child_index(Node* tree, NNEvaluation* evaluation);

    Mcts* const mcts_;
    const Params& params_;
    const SearchParams* search_params_ = nullptr;
    std::thread* thread_ = nullptr;
    const int thread_id_;
  };

  struct PUCTStats {
    static constexpr float eps = 1e-6;  // needed when N == 0
    using PVec = LocalPolicyProbDistr;

    PUCTStats(const Params& params, const SearchParams& search_params, const Node* tree);

    seat_index_t cp;
    const PVec& P;
    PVec V;
    PVec N;
    PVec VN;
    PVec E;
    PVec PUCT;
  };

  using search_thread_vec_t = std::vector<SearchThread*>;

  /*
   * The NNEvaluationService services multiple search threads, which may belong to multiple Mcts instances (if two
   * Mcts agents are playing against each other for instance).
   *
   * The main API is the evaluate() method. It tensorizes a game state, passes the tensor to a neural network, and
   * returns the output. Under its hood, it batches multiple evaluate() requests in order to maximize GPU throughput.
   * This batching is transparent to the caller, as the evaluate() method blocks internally until the batch is full
   * (or until a timeout is hit).
   *
   * Batching of N evaluations is accomplished by maintaining a length-N array of evaluation objects, and various
   * tensors (for nnet input and output) of shape (N, ...). Each evaluate() call gets assigned a particular index i
   * with 0 <= i < N, and writes to the i'th slot of these data structures. A separate evaluation thread issues the
   * nnet evaluation and writes to the i'th slot of the output data structures.
   *
   * The service has an LRU cache, which helps to avoid the costly GPU operations when possible.
   *
   * Here is a detailed description of how this implementation handles the various thread safety considerations.
   *
   * There are three mutexes:
   *
   * - cache_mutex_: prevents race-conditions on cache reads/writes - especially important because without locking,
   *                 cache eviction can lead to a key-value pair disappearing after checking for the key
   * - batch_data_.mutex: prevents race-conditions on reads/writes of batch_data_
   * - batch_metadata_.mutex: prevents race-conditions on reads/writes of batch_metadata_
   *
   * The batch_data_ member consists of:
   *
   * - input: the batch input tensor
   * - value/policy: the batch output tensors
   * - eval_ptr_data: mainly an array of N smart-pointers to a struct that has copied a slice of the value/policy
   *                  tensors.
   *
   * The batch_metadata_ member consists of three ints:
   *
   * - reserve_index: the next slot of batch_data_.input to write to
   * - commit_count: the number of slots of batch_data_.input that have been written to
   * - unread_count: the number of entries of batch_data_.eval_ptr_data that have not yet been read by their
   *                 corresponding search threads
   *
   * The loop() and evaluate() methods of NNEvaluationService have been carefully written to ensure that the reads
   * and writes of these data structures are thread-safe.
   *
   * Compiling with -DMCTS_THREADING_DEBUG will enable a bunch of prints that allow you to watch the sequence of
   * operations in the interleaving threads.
   */
  class NNEvaluationService {
  public:
    struct Request {
      SearchThread* thread;
      Node* tree;
      symmetry_index_t sym_index;
    };

    struct Response {
      NNEvaluation_sptr ptr;
      bool used_cache;
    };

    /*
     * Constructs an evaluation thread and returns it.
     *
     * If another thread with the given model_filename has already been create()'d, then returns that. If that returned
     * thread does not match the thread parameters (batch_size, nn_eval_timeout_ns, cache_size), then raises an
     * exception.
     */
    static NNEvaluationService* create(const Mcts* mcts);

    /*
     * Instantiates the thread_ member if not yet instantiated. This spawns a new thread.
     *
     * If the thread_ member is already instantiated, then this is a no-op.
     */
    void connect();

    void disconnect();

    /*
     * Called by search threads. Returns immediately if we get a cache-hit. Otherwise, blocks on the service thread.
     *
     * Note that historically, parallel MCTS did evaluations asynchronously. AlphaGo Zero was the first version that
     * switched to blocking evaluations.
     *
     * "Compared to the MCTS in AlphaGo Fan and AlphaGo Lee, the principal differences are...each search thread simply
     * waits for the neural network evaluation, rather than performing evaluation and backup asynchronously"
     *
     * - Mastering the Game of Go without Human Knowledge (page 27)
     *
     * https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
     */
    Response evaluate(const Request&);

    void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;

    int64_t evaluated_positions() const { return evaluated_positions_; }
    int64_t batches_evaluated() const { return batches_evaluated_; }
    float avg_batch_size() const { return evaluated_positions() * 1.0 / std::max(int64_t(1), batches_evaluated()); }

    void record_puct_calc(bool virtual_loss_influenced) {
      this->total_puct_calcs_++;
      if (virtual_loss_influenced) {
        this->virtual_loss_influenced_puct_calcs_++;
      }
    }

    static float pct_virtual_loss_influenced_puct_calcs();  // summed over all instances
    static float global_avg_batch_size();  // averaged over all instances

  private:
    using instance_map_t = std::map<boost::filesystem::path, NNEvaluationService*>;
    using cache_key_t = StateEvaluationKey<GameState>;
    using cache_t = util::LRUCache<cache_key_t, NNEvaluation_asptr>;

    NNEvaluationService(const boost::filesystem::path& net_filename, int batch_size_limit,
                        std::chrono::nanoseconds timeout_duration, size_t cache_size,
                        const boost::filesystem::path& profiling_dir);
    ~NNEvaluationService();

    void batch_evaluate();
    void loop();

    Response check_cache(SearchThread* thread, const cache_key_t& cache_key);
    void wait_until_batch_reservable(SearchThread* thread, std::unique_lock<std::mutex>& metadata_lock);
    int allocate_reserve_index(SearchThread* thread, std::unique_lock<std::mutex>& metadata_lock);
    void tensorize_and_transform_input(const Request& request, const cache_key_t& cache_key, int reserve_index);
    void increment_commit_count(SearchThread* thread);
    NNEvaluation_sptr get_eval(SearchThread* thread, int reserve_index, std::unique_lock<std::mutex>& metadata_lock);
    void wait_until_all_read(SearchThread* thread, std::unique_lock<std::mutex>& metadata_lock);

    void wait_until_batch_ready();
    void wait_for_first_reservation();
    void wait_for_last_reservation();
    void wait_for_commits();

    bool active() const { return num_connections_; }

    struct eval_ptr_data_t {
      NNEvaluation_asptr eval_ptr;

      cache_key_t cache_key;
      ActionMask valid_actions;
      SymmetryTransform* transform;
    };

    enum region_t {
      kWaitingUntilBatchReady = 0,
      kWaitingForFirstReservation = 1,
      kWaitingForLastReservation = 2,
      kWaitingForCommits = 3,
      kCopyingCpuToGpu = 4,
      kEvaluatingNeuralNet = 5,
      kCopyingToPool = 6,
      kAcquiringCacheMutex = 7,
      kFinishingUp = 8,
      kNumRegions = 9
    };

    using profiler_t = util::Profiler<int(kNumRegions), kEnableVerboseProfiling>;

    struct profiling_stats_t {
      time_point_t start_times[kNumRegions + 1];
      int batch_size;
    };

    void record_for_profiling(region_t region);
    void dump_profiling_stats();

#ifdef PROFILE_MCTS
    profiler_t* get_profiler() { return &profiler_; }
    FILE* get_profiling_file() const { return profiling_file_; }
    const char* get_profiler_name() const { return profiler_name_.c_str(); }
    void init_profiling(const char* filename, const char* name) {
      profiling_file_ = fopen(filename, "w");
      profiler_name_ = name;
      profiler_.skip_next_n_dumps(5);
    }
    void close_profiling_file() { fclose(profiling_file_); }

    profiler_t profiler_;
    std::string profiler_name_;
    FILE* profiling_file_ = nullptr;
    int profile_count_ = 0;
#else  // PROFILE_MCTS
    constexpr profiler_t* get_profiler() { return nullptr; }
    static FILE* get_profiling_file() { return nullptr; }
    const char* get_profiler_name() const { return nullptr; }
    void init_profiling(const char* filename, const char* name) {}
    void close_profiling_file() {}
#endif  // PROFILE_MCTS

    static instance_map_t instance_map_;
    static int next_instance_id_;

    const int instance_id_;  // for naming debug/profiling output files

    std::thread* thread_ = nullptr;
    std::mutex cache_mutex_;
    std::mutex connection_mutex_;

    std::condition_variable cv_service_loop_;
    std::condition_variable cv_evaluate_;

    NeuralNet net_;
    struct batch_data_t {
      batch_data_t(int batch_size);

      std::mutex mutex;
      FullPolicyTensor policy;
      FullValueTensor value;
      FullInputTensor input;
      eval_ptr_data_t* eval_ptr_data;
    };
    batch_data_t batch_data_;

    common::NeuralNet::input_vec_t input_vec_;
    torch::Tensor torch_input_gpu_;
    cache_t cache_;

    const std::chrono::nanoseconds timeout_duration_;
    const int batch_size_limit_;

    time_point_t deadline_;
    struct batch_metadata_t {
      std::mutex mutex;
      int reserve_index = 0;
      int commit_count = 0;
      int unread_count = 0;
      bool accepting_reservations = true;
      std::string repr() const {
        return util::create_string("res=%d, com=%d, unr=%d, acc=%d",
                                   reserve_index, commit_count, unread_count, accepting_reservations);
      }
    };
    batch_metadata_t batch_metadata_;

    int num_connections_ = 0;

    std::atomic<int> cache_hits_ = 0;
    std::atomic<int> cache_misses_ = 0;
    std::atomic<int64_t> evaluated_positions_ = 0;
    std::atomic<int64_t> batches_evaluated_ = 0;
    std::atomic<int64_t> total_puct_calcs_ = 0;
    std::atomic<int64_t> virtual_loss_influenced_puct_calcs_ = 0;
  };

  class NodeReleaseService {
  public:
    struct work_unit_t {
      work_unit_t(Node* n, Node* a) : node(n), arg(a) {}

      Node* node;
      Node* arg;
    };

    static void release(Node* node, Node* arg=nullptr) { instance_.release_helper(node, arg); }

  private:
    NodeReleaseService();
    ~NodeReleaseService();

    void loop();
    void release_helper(Node* node, Node* arg);

    static NodeReleaseService instance_;

    using work_queue_t = std::vector<work_unit_t>;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    work_queue_t work_queue_[2];
    int queue_index_ = 0;
    int release_count_ = 0;
    int max_queue_size_ = 0;
    bool destructing_ = false;
  };

public:
  /*
   * In multi-threaded mode, the search threads can continue running outside of the main search() method. For example,
   * when playing against a human player, we can continue growing the MCTS tree while the human player thinks.
   */
  static constexpr int kDefaultMaxTreeSize =  4096;

  static int next_instance_id_;  // for naming debug/profiling output files

  Mcts(const Params& params);
  ~Mcts();

  int instance_id() const { return instance_id_; }
  const Params& params() const { return params_; }
  int num_search_threads() const { return params_.num_search_threads; }
  bool search_active() const { return search_active_; }
  NNEvaluationService* nn_eval_service() const { return nn_eval_service_; }

  void start();
  void clear();
  void receive_state_change(seat_index_t, const GameState&, action_index_t);
  const MctsResults* search(const Tensorizor& tensorizor, const GameState& game_state, const SearchParams& params);
  void add_dirichlet_noise(LocalPolicyProbDistr& P);
  float root_softmax_temperature() const { return root_softmax_temperature_.value(); }

  void start_search_threads(const SearchParams* search_params);
  void wait_for_search_threads();
  void stop_search_threads();
  void run_search(SearchThread* thread, int tree_size_limit);
  void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;
  float avg_batch_size() const { return nn_eval_service_->avg_batch_size(); }
  static float global_avg_batch_size() { return NNEvaluationService::global_avg_batch_size(); }
  void record_puct_calc(bool virtual_loss_influenced) { if (nn_eval_service_) nn_eval_service_->record_puct_calc(virtual_loss_influenced); }

  static float pct_virtual_loss_influenced_puct_calcs() { return NNEvaluationService::pct_virtual_loss_influenced_puct_calcs(); }

#ifdef PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return boost::filesystem::path(params_.profiling_dir); }
#else  // PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return {}; }
#endif  // PROFILE_MCTS

private:
  void prune_counts(const SearchParams&);
  static void init_profiling_dir(const std::string& profiling_dir);

  eigen_util::UniformDirichletGen<float> dirichlet_gen_;
  Eigen::Rand::P8_mt19937_64 rng_;

  const Params params_;
  const SearchParams offline_search_params_;
  const int instance_id_;
  math::ExponentialDecay root_softmax_temperature_;
  search_thread_vec_t search_threads_;
  NNEvaluationService* nn_eval_service_ = nullptr;

  Node* root_ = nullptr;
  MctsResults results_;

  std::mutex search_mutex_;
  std::condition_variable cv_search_;
  int num_active_search_threads_ = 0;
  bool search_active_ = false;
  bool connected_ = false;
};

}  // namespace common

#include <common/inl/Mcts.inl>
