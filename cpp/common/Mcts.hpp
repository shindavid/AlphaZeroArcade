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
#include <util/Profiler.hpp>

namespace common {

/*
 * TODO: move the various inner-classes of Mcts into separate files as standalone-classes.
 *
 * TODO: use CRTP for slightly more elegant inheritance mechanics.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
class Mcts {
private:
  class SearchThread;

public:
  static constexpr bool kEnableProfiling = IS_MACRO_ASSIGNED_TO_1(PROFILE_MCTS);
  static constexpr bool kEnableVerboseProfiling = IS_MACRO_ASSIGNED_TO_1(PROFILE_MCTS_VERBOSE);

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameState::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using TensorizorTypes = common::TensorizorTypes<Tensorizor>;
  using GameStateTypes = common::GameStateTypes<GameState>;

  using MctsResults = common::MctsResults<GameState>;
  using SymmetryTransform = AbstractSymmetryTransform<GameState, Tensorizor>;
  using ValueProbDistr = typename GameStateTypes::ValueProbDistr;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ActionMask = typename GameStateTypes::ActionMask;
  using LocalPolicyLogitDistr = typename GameStateTypes::LocalPolicyLogitDistr;
  using LocalPolicyProbDistr = typename GameStateTypes::LocalPolicyProbDistr;
  using GlobalPolicyCountDistr = typename GameStateTypes::GlobalPolicyCountDistr;

  using FullInputTensor = typename TensorizorTypes::DynamicInputTensor;
  using FullValueArray = typename GameStateTypes::template ValueArray<Eigen::Dynamic>;
  using FullPolicyArray = typename GameStateTypes::template PolicyArray<Eigen::Dynamic>;
  using ValueArray1D = typename GameStateTypes::ValueArray1D;
  using PolicyArray1D = typename GameStateTypes::PolicyArray1D;

  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

  /*
   * Params pertains to a single Mcts instance.
   *
   * By contrast, SimParams pertains to each individual sim() call.
   */
  struct Params {
    auto make_options_description();

    std::string nnet_filename;
    bool uniform_model = false;
    int num_search_threads = 8;
    int batch_size_limit = 144;
    bool run_offline = false;
    int offline_tree_size_limit = 4096;
    int64_t nn_eval_timeout_ns = util::us_to_ns(250);
    size_t cache_size = 1048576;

    float root_softmax_temperature = 1.03;
    float cPUCT = 1.1;
    float dirichlet_mult = 0.25;
    float dirichlet_alpha = 0.03;
    bool disable_eliminations = true;
    bool speculative_evals = false;
#ifdef PROFILE_MCTS
    std::string profiling_dir;
#endif  // PROFILE_MCTS
  };

  /*
   * SimParams pertain to a single call to sim(). Even given a single Mcts instance, different sim() calls can have
   * different SimParams. For instance, for KataGo, there are "fast" searches and "full" searches, which differ
   * in their tree_size_limit and dirchlet settings.
   *
   * By contrast, Params pertains to a single Mcts instance.
   */
  struct SimParams {
    static SimParams make_offline_params(int limit) { return SimParams{limit, true}; }

    int tree_size_limit = 100;
    bool disable_noise = false;
  };

private:
  class NNEvaluation {
  public:
    NNEvaluation(const ValueArray1D& value, const PolicyArray1D& policy, const ActionMask& valid_actions);
    const ValueProbDistr& value_prob_distr() const { return value_prob_distr_; }
    const LocalPolicyProbDistr& local_policy_logit_distr() const { return local_policy_logit_distr_; }

  protected:
    ValueProbDistr value_prob_distr_;
    LocalPolicyLogitDistr local_policy_logit_distr_;
  };
  using NNEvaluation_sptr = std::shared_ptr<NNEvaluation>;
  using NNEvaluation_asptr = util::AtomicSharedPtr<NNEvaluation>;

  /*
   * A Node consists of n=4 main groups of non-const member variables:
   *
   * lazily_initialized_data_: state + action -> state', computed lazily since not immediately needed upon child expand
   * children_data_: the addresses/number of children nodes, needed for tree traversal
   * evaluation_data_: policy/value vectors that come from neural net evaluation
   * stats_: values that get updated throughout MCTS via backpropagation
   *
   * Of these, only stats_ are continuously changing. The others are written only once. They are non-const in the
   * sense that they are lazily written, after-object-construction.
   *
   * During MCTS, multiple search threads will try to read and write these values. Thread-safety is achieved in a
   * high-performance manner through a carefully orchestrated combination of mutexes, condition-variables, and
   * lockfree mechanisms.
   *
   * NAMING NOTE: Methods with a leading underscore are NOT thread-safe. Such methods are expected to be called in
   * a context that guarantees the appropriate level of thread-safety.
   */
  class Node {
  public:
    enum evaluation_state_t : int8_t {
      kUnset,
      kPending,
      kSet,
    };

    Node(Node* parent, action_index_t action);
    Node(const Tensorizor&, const GameState&, const GameOutcome&, bool disable_noise);
    Node(const Node& node, bool prune_parent=false);

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
    void _release(Node* protected_child=nullptr);

    /*
     * Set child->parent = this for all children of this.
     *
     * This is the only reason that stable_data_ is not const.
     */
    void _adopt_children();

    std::condition_variable& cv_evaluate_and_expand() { return cv_evaluate_and_expand_; }
    std::mutex& lazily_initialized_data_mutex() { return lazily_initialized_data_mutex_; }
    std::mutex& evaluation_data_mutex() { return evaluation_data_mutex_; }
    std::mutex& stats_mutex() { return stats_mutex_; }

    GlobalPolicyCountDistr get_effective_counts() const;
    void backprop(const ValueProbDistr& value);
    void backprop_with_virtual_undo(const ValueProbDistr& value);
    void virtual_backprop();
    void perform_eliminations(const ValueProbDistr& outcome);
    ValueArray1D make_virtual_loss() const;
    void mark_as_fully_analyzed();

    void _lazy_init();
    void _expand_children();

    action_index_t action() const { return stable_data_.action_; }
    Node* parent() const { return stable_data_.parent_; }
    bool is_root() const { return !stable_data_.parent_; }
    bool disable_noise() const { return stable_data_.disable_noise_; }

    const Tensorizor& _tensorizor() const { return lazily_initialized_data_.union_.data_.tensorizor_; }
    const GameState& _state() const { return lazily_initialized_data_.union_.data_.state_; }
    const GameOutcome& _outcome() const { return lazily_initialized_data_.union_.data_.outcome_; }
    symmetry_index_t _sym_index() const { return lazily_initialized_data_.union_.data_.sym_index_; }
    player_index_t _current_player() const { return lazily_initialized_data_.union_.data_.current_player_; }
    const ActionMask& _valid_action_mask() const { return lazily_initialized_data_.union_.data_.valid_action_mask_; }
    bool _lazily_initialized() const { return lazily_initialized_data_.initialized_; }

    bool _has_children() const { return children_data_.num_children_unsafe(); }
    int _num_children() const { return children_data_.num_children_unsafe(); }
    Node* _get_child(int c) const { return children_data_.first_child_unsafe() + c; }
    Node* _find_child(action_index_t action) const;

    const auto& _value_avg() const { return stats_.value_avg_; }
    bool _eliminated() const { return stats_.eliminated_; }
    float _V_floor(player_index_t p) const { return stats_.V_floor_(p); }
    float _effective_value_avg(player_index_t p) const { return stats_.effective_value_avg_(p); }
    int _effective_count() const { return stats_.eliminated_ ? 0 : stats_.count_; }
    bool _has_certain_outcome() const { return stats_.V_floor_.sum() > 1 - 1e-6; }  // 1e-6 fudge factor for floating-point error
    bool _can_be_eliminated() const { return stats_.V_floor_.maxCoeff() == 1; }  // won/lost positions, not drawn ones

    const ActionMask& _fully_analyzed_action_mask() const { return evaluation_data_.fully_analyzed_actions_; }
    const LocalPolicyProbDistr& _local_policy_prob_distr() const { return evaluation_data_.local_policy_prob_distr_; }
    void _set_local_policy_prob_distr(const LocalPolicyProbDistr& distr) {
      evaluation_data_.local_policy_prob_distr_ = distr;
    }
    NNEvaluation_sptr _evaluation() const { return evaluation_data_.ptr_.load(); }
    void _set_evaluation(NNEvaluation_sptr eval) { evaluation_data_.ptr_.store(eval); }
    evaluation_state_t _evaluation_state() const { return evaluation_data_.state_; }
    void _set_evaluation_state(evaluation_state_t state) { evaluation_data_.state_ = state; }

  private:
    float _get_max_V_floor_among_children(player_index_t p, Node* first_child, int num_children) const;
    float _get_min_V_floor_among_children(player_index_t p, Node* first_child, int num_children) const;

    struct stable_data_t {
      stable_data_t(Node* parent, action_index_t action, bool disable_noise);
      stable_data_t(const stable_data_t& data, bool prune_parent);

      Node* parent_;
      action_index_t action_;
      bool disable_noise_;
    };

    struct lazily_initialized_data_t {
      struct data_t {
        data_t(Node* parent, action_index_t action);
        data_t(const Tensorizor&, const GameState&, const GameOutcome&);

        Tensorizor tensorizor_;
        GameState state_;
        GameOutcome outcome_;
        ActionMask valid_action_mask_;
        player_index_t current_player_;
        symmetry_index_t sym_index_;
      };

      union union_t {
        union_t() : dummy_(false) {}
        union_t(const union_t& u) : data_(u.data_) {}
        union_t(Node* parent, action_index_t action) : data_(parent, action) {}
        union_t(const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome)
            : data_(tensorizor, state, outcome) {}

        data_t data_;
        bool dummy_;
      };

      lazily_initialized_data_t() = default;
      lazily_initialized_data_t(Node* parent, action_index_t action)
        : union_(parent, action)
        , initialized_(true) {}
      lazily_initialized_data_t(const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome)
        : union_(tensorizor, state, outcome)
        , initialized_(true) {}

      union_t union_;
      bool initialized_ = false;
    };

    /*
     * Writers must FIRST write num_children_, and THEN first_child_.
     *
     * Readers should ignore num_children_ if first_child_ is null.
     *
     * Following this discipline allows us to use children_data_t in a lockfree manner.
     *
     * We use the volatile keyword here to ensure that read/write order is not changed by the compiler.
     *
     * See: https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf
     */
    struct children_data_t {
      void write(Node* first_child, int num_children) {
        num_children_ = num_children;
        first_child_ = first_child;
      }

      void read(Node** first_child, int* num_children) const {
        *first_child = const_cast<Node*>(first_child_);
        *num_children = (*first_child) ? num_children_ : 0;
      }

      Node* first_child_unsafe() const { return const_cast<Node*>(first_child_); }  // read() is safer, be careful
      int num_children_unsafe() const { return num_children_; }  // read() is safer, be careful

    private:
      volatile Node* first_child_ = nullptr;
      volatile int num_children_ = 0;
    };

    struct evaluation_data_t {
      evaluation_data_t() = default;
      evaluation_data_t(const ActionMask& valid_actions);

      NNEvaluation_asptr ptr_;
      LocalPolicyProbDistr local_policy_prob_distr_;
      evaluation_state_t state_ = kUnset;
      ActionMask fully_analyzed_actions_;  // means that every leaf descendent is a terminal game state
    };

    struct stats_t {
      stats_t();

      ValueArray1D value_avg_;
      ValueArray1D effective_value_avg_;
      ValueArray1D V_floor_;
      int count_ = 0;
      bool eliminated_ = false;
    };

    std::condition_variable cv_evaluate_and_expand_;
    mutable std::mutex lazily_initialized_data_mutex_;
    mutable std::mutex evaluation_data_mutex_;
    mutable std::mutex stats_mutex_;
    stable_data_t stable_data_;  // effectively const
    lazily_initialized_data_t lazily_initialized_data_;
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
    void launch(const SimParams* sim_params);
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
      bool performed_expansion;
    };

    void lazily_init(Node* tree);
    void backprop_outcome(Node* tree, const ValueProbDistr& outcome);
    void perform_eliminations(Node* tree, const ValueProbDistr& outcome);
    void mark_as_fully_analyzed(Node* tree);
    evaluate_and_expand_result_t evaluate_and_expand(Node* tree, bool speculative);
    void evaluate_and_expand_unset(
        Node* tree, std::unique_lock<std::mutex>* lock, evaluate_and_expand_result_t* data, bool speculative);
    void evaluate_and_expand_pending(Node* tree, std::unique_lock<std::mutex>* lock);
    void expand_children(Node* tree);

    /*
     * Used in visit().
     *
     * Applies PUCT criterion to select the best child to visit from the given Node.
     *
     * TODO: as we experiment with things like auxiliary NN output heads, dynamic cPUCT values, etc., this method will
     * evolve. It probably makes sense to have the behavior as part of the Tensorizor, since there is coupling with NN
     * architecture (in the form of output heads).
     */
    Node* get_best_child(Node* tree, NNEvaluation* evaluation);

    Mcts* const mcts_;
    const Params& params_;
    const SimParams* sim_params_ = nullptr;
    std::thread* thread_ = nullptr;
    const int thread_id_;
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
   * When the number of search threads is 1, we simply do everything in the main thread, mainly for easier debugging.
   * When we have more than 1 search thread, we encounter sensitive thread-safety considerations. Here is a detailed
   * description of how this implementation handles them.
   *
   * There are two mutexes:
   *
   * - cache_mutex_: prevents race-conditions on cache reads/writes - especially important because without locking,
   *                 cache eviction can lead to a key-value pair disappearing after checking for the key
   *
   * - batch_mutex_: prevents race-conditions on batch reads/writes
   *
   * There are three separate members that are used to synchronize batch writing, all protected by batch_mutex_:
   *
   * - batch_reserve_index_: keeps track of the next batch slot to write to. A search thread does a protected
   *                         read + increment of this member, and then with the mutex released, starts the work of
   *                         tensorizing the game state and writing to the appropriate slot.
   *
   * - batch_commit_count_: Once the search thread has finished tensorizing and writing to its slot, it increments
   *                        the batch_commit_count_. This allows the service thread to detect when it is safe to
   *                        evaluate the batch.
   *
   * - batch_unread_count_: After the service thread has written the N nnet outputs, the search threads can start
   *                        reading those outputs. Without care, a race condition could cause eager search threads to
   *                        overwrite that data before the prior threads fully read that data. This unread count
   *                        value helps to prevent this race condition.
   *
   * Note that batch_reserve_index_ and batch_commit_count_ could have been rolled into one single count. This would
   * have been simpler, but increased the duration for which batch_mutex_ is held (as the tensorizing would then
   * need to happen under the mutex). Separation allows tensorization to occur outside any mutex locks.
   *
   * Search threads will detect that the batch is fully saturated by checking batch_reserve_index_, and wait until it
   * is reset by the evaluation thread before proceeding with tensorization.
   *
   * The evaluation thread will detect that search threads are mid-writing by comparing batch_commit_count_ to
   * batch_reserve_index_. Only when they are equal will it proceed to query the GPU and write to the output slots.
   */
  class NNEvaluationService {
  public:
    struct Request {
      SearchThread* thread;
      const Tensorizor* tensorizor;
      const GameState* state;
      const ActionMask* valid_action_mask;
      symmetry_index_t sym_index;
    };

    struct Response {
      NNEvaluation_sptr ptr;
      bool used_cache;
    };

    /*
     * Constructs an evaluation thread and returns it.
     *
     * If another thread with the given nnet_filename has already been create()'d, then returns that. If that returned
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

    static float global_avg_batch_size();  // averaged over all instances

  private:
    NNEvaluationService(const boost::filesystem::path& net_filename, int batch_size_limit,
                        std::chrono::nanoseconds timeout_duration, size_t cache_size,
                        const boost::filesystem::path& profiling_dir);
    ~NNEvaluationService();

    void batch_evaluate();
    void loop();

    bool active() const { return num_connections_; }
    bool all_batch_reservations_committed() const { return batch_reserve_index_ == batch_commit_count_; }
    bool batch_reservations_full() const { return batch_reserve_index_ == batch_size_limit_; }
    bool batch_reservations_empty() const { return batch_reserve_index_ == 0; }
    bool batch_reservable() const { return batch_unread_count_ == 0 && batch_reserve_index_ < batch_size_limit_; }

    using instance_map_t = std::map<boost::filesystem::path, NNEvaluationService*>;
    using cache_key_t = StateEvaluationKey<GameState>;
    using cache_t = util::LRUCache<cache_key_t, NNEvaluation_asptr>;

    struct eval_ptr_data_t {
      NNEvaluation_asptr eval_ptr;

      cache_key_t cache_key;
      ActionMask valid_actions;
      SymmetryTransform* transform;
    };

    enum region_t {
      kAcquiringBatchMutex = 0,
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
    std::mutex batch_mutex_;
    std::condition_variable cv_service_loop_;
    std::condition_variable cv_evaluate_;

    NeuralNet net_;
    FullPolicyArray policy_batch_;
    FullValueArray value_batch_;
    FullInputTensor input_batch_;
    eval_ptr_data_t* evaluation_data_batch_;

    common::NeuralNet::input_vec_t input_vec_;
    torch::Tensor torch_input_gpu_;
    cache_t cache_;

    const std::chrono::nanoseconds timeout_duration_;
    const int batch_size_limit_;

    time_point_t deadline_;
    int batch_reserve_index_ = 0;
    int batch_commit_count_ = 0;
    int batch_unread_count_ = 0;

    int num_connections_ = 0;

    int cache_hits_ = 0;
    int cache_misses_ = 0;
    int64_t evaluated_positions_ = 0;
    int64_t batches_evaluated_ = 0;
  };

  class NodeReleaseService {
  public:
    struct work_unit_t {
      work_unit_t(Node* n, Node* a) : node(n), arg(a) {}

      Node* node;
      Node* arg;
    };

    static void release(Node* node, Node* arg=nullptr) { instance_._release(node, arg); }

  private:
    NodeReleaseService();
    ~NodeReleaseService();

    void loop();
    void _release(Node* node, Node* arg);

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
   * In multi-threaded mode, the search threads can continue running outside of the main sim() method. For example,
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
  void receive_state_change(player_index_t, const GameState&, action_index_t, const GameOutcome&);
  const MctsResults* sim(const Tensorizor& tensorizor, const GameState& game_state, const SimParams& params);
  void add_dirichlet_noise(LocalPolicyProbDistr& P);

  void start_search_threads(const SimParams* sim_params);
  void wait_for_search_threads();
  void stop_search_threads();
  void run_search(SearchThread* thread, int tree_size_limit);
  void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;
  float avg_batch_size() const { return nn_eval_service_->avg_batch_size(); }
  static float global_avg_batch_size() { return NNEvaluationService::global_avg_batch_size(); }

#ifdef PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return boost::filesystem::path(params_.profiling_dir); }
#else  // PROFILE_MCTS
  boost::filesystem::path profiling_dir() const { return {}; }
#endif  // PROFILE_MCTS

private:
  static void init_profiling_dir(const std::string& profiling_dir);

  eigen_util::UniformDirichletGen<float> dirichlet_gen_;
  Eigen::Rand::P8_mt19937_64 rng_;

  const Params params_;
  const SimParams offline_sim_params_;
  const int instance_id_;
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
