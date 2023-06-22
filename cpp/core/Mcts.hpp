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

#include <core/AbstractSymmetryTransform.hpp>
#include <core/BasicTypes.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/MctsResults.hpp>
#include <core/NeuralNet.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/Node.hpp>
#include <util/AtomicSharedPtr.hpp>
#include <util/BitSet.hpp>
#include <util/BoostUtil.hpp>
#include <util/CppUtil.hpp>
#include <util/LRUCache.hpp>
#include <util/Math.hpp>
#include <util/Profiler.hpp>

namespace core {

/*
 * TODO: move the various inner-classes of Mcts into separate files as standalone-classes. Proposed class-renaming:
 *
 * core::Mcts<GameState, Tensorizor> -> mcts::Tree<GameState, Tensorizor>
 * core::Mcts<GameState, Tensorizor>::Params -> mcts::Params<GameState, Tensorizor>
 * core::Mcts<GameState, Tensorizor>::SearchThread -> mcts::SearchThread<GameState, Tensorizor>
 * core::Mcts<GameState, Tensorizor>::Node -> mcts::Node<GameState, Tensorizor>
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
  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using NNEvaluation_asptr = typename NNEvaluation::asptr;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;
  using Node = mcts::Node<GameState, Tensorizor>;

  using TensorizorTypes = core::TensorizorTypes<Tensorizor>;
  using GameStateTypes = core::GameStateTypes<GameState>;

  static constexpr int kNumPlayers = GameState::kNumPlayers;
  static constexpr int kNumGlobalActions = GameStateTypes::kNumGlobalActions;
  static constexpr int kMaxNumLocalActions = GameState::kMaxNumLocalActions;

  using dtype = typename GameStateTypes::dtype;
  using child_index_t = int;

  using MctsResults = core::MctsResults<GameState>;
  using SymmetryTransform = AbstractSymmetryTransform<GameState, Tensorizor>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using ActionMask = typename GameStateTypes::ActionMask;

  using PolicyArray = typename GameStateTypes::PolicyArray;
  using ValueArray = typename GameStateTypes::ValueArray;
  using LocalPolicyArray = typename GameStateTypes::LocalPolicyArray;

  using InputTensor = typename TensorizorTypes::InputTensor;
  using PolicyTensor = typename GameStateTypes::PolicyTensor;
  using ValueTensor = typename GameStateTypes::ValueTensor;

  using InputShape = typename TensorizorTypes::InputShape;
  using PolicyShape = typename GameStateTypes::PolicyShape;
  using ValueShape = typename GameStateTypes::ValueShape;

  using InputScalar = torch_util::convert_type_t<typename InputTensor::Scalar>;
  using PolicyScalar = torch_util::convert_type_t<typename PolicyTensor::Scalar>;
  using ValueScalar = torch_util::convert_type_t<typename ValueTensor::Scalar>;

  using InputFloatTensor = Eigen::TensorFixedSize<dtype, InputShape, Eigen::RowMajor>;
  using DynamicInputFloatTensor = Eigen::Tensor<dtype, InputShape::count + 1, Eigen::RowMajor>;

  using player_bitset_t = std::bitset<kNumPlayers>;

  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

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
    std::string cuda_device = "cuda:0";
    int num_search_threads = 8;
    int batch_size_limit = 216;
    bool enable_pondering = false;  // pondering = think during opponent's turn
    int pondering_tree_size_limit = 4096;
    int64_t nn_eval_timeout_ns = util::us_to_ns(250);
    size_t cache_size = 1048576;

    std::string root_softmax_temperature_str;
    float cPUCT = 1.1;
    float cFPU = 0.2;
    float dirichlet_mult = 0.25;
    float dirichlet_alpha_sum = 0.03 * 361;
    bool disable_eliminations = true;
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
    static SearchParams make_pondering_params(int limit) { return SearchParams{limit, true, true}; }

    int tree_size_limit = 100;
    bool disable_exploration = false;
    bool ponder = false;
  };

private:
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
    bool is_pondering() const { return search_params_->ponder; }

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

    using profiler_t = util::Profiler<int(kNumRegions), mcts::kEnableVerboseProfiling>;

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

    void backprop_outcome(Node* tree, const ValueArray& outcome);
    void perform_eliminations(Node* tree, const ValueArray& outcome);
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
    using PVec = LocalPolicyArray;

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
    void record_puct_calc(bool virtual_loss_influenced);

    static void end_session();
    static float pct_virtual_loss_influenced_puct_calcs();  // summed over all instances

  private:
    using instance_map_t = std::map<boost::filesystem::path, NNEvaluationService*>;
    using cache_key_t = StateEvaluationKey<GameState>;
    using cache_t = util::LRUCache<cache_key_t, NNEvaluation_asptr>;

    NNEvaluationService(const boost::filesystem::path& net_filename, const std::string& cuda_device,
                        int batch_size_limit, std::chrono::nanoseconds timeout_duration, size_t cache_size,
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

    using profiler_t = util::Profiler<int(kNumRegions), mcts::kEnableVerboseProfiling>;

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
    static bool session_ended_;

    const int instance_id_;  // for naming debug/profiling output files

    std::thread* thread_ = nullptr;
    std::mutex cache_mutex_;
    std::mutex connection_mutex_;

    std::condition_variable cv_service_loop_;
    std::condition_variable cv_evaluate_;

    NeuralNet net_;

    struct tensor_group_t {
      void load_output_from(int row, torch::Tensor& torch_policy, torch::Tensor& torch_value);

      InputTensor input;
      PolicyTensor policy;
      ValueTensor value;
      seat_index_t current_player;
      eval_ptr_data_t eval_ptr_data;
    };

    struct batch_data_t {
      batch_data_t(int batch_size);
      ~batch_data_t();
      void copy_input_to(int num_rows, DynamicInputFloatTensor& full_input);

      std::mutex mutex;
      tensor_group_t* tensor_groups_;
    };
    batch_data_t batch_data_;

    core::NeuralNet::input_vec_t input_vec_;
    torch::Tensor torch_input_gpu_;
    torch::Tensor torch_policy_;
    torch::Tensor torch_value_;
    DynamicInputFloatTensor full_input_;
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
  void add_dirichlet_noise(LocalPolicyArray& P);
  float root_softmax_temperature() const { return root_softmax_temperature_.value(); }

  void start_search_threads(const SearchParams* search_params);
  void wait_for_search_threads();
  void stop_search_threads();
  void run_search(SearchThread* thread, int tree_size_limit);
  void get_cache_stats(int& hits, int& misses, int& size, float& hash_balance_factor) const;
  void record_puct_calc(bool virtual_loss_influenced) { if (nn_eval_service_) nn_eval_service_->record_puct_calc(virtual_loss_influenced); }

  static float pct_virtual_loss_influenced_puct_calcs() { return NNEvaluationService::pct_virtual_loss_influenced_puct_calcs(); }
  static void end_session() { NNEvaluationService::end_session(); }

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
  const SearchParams pondering_search_params_;
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

}  // namespace core

#include <core/inl/Mcts.inl>