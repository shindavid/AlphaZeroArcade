#pragma once

#include <core/CmdServerClient.hpp>
#include <core/CmdServerListener.hpp>
#include <core/DerivedTypes.hpp>
#include <core/GameStateConcept.hpp>
#include <core/NeuralNet.hpp>
#include <core/PerfStats.hpp>
#include <core/TensorizorConcept.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationServiceParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/SharedData.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/HashablePair.hpp>
#include <util/LRUCache.hpp>
#include <util/TorchUtil.hpp>

#include <vector>

namespace mcts {

/*
 * The NNEvaluationService services multiple search threads, which may belong to multiple
 * mcts::Manager instances (if two mcts agents are playing against each other for instance).
 *
 * The main API is the evaluate() method. It tensorizes a game state, passes the tensor to a neural
 * network, and returns the output. Under its hood, it batches multiple evaluate() requests in order
 * to maximize GPU throughput. This batching is transparent to the caller, as the evaluate() method
 * blocks internally until the batch is full (or until a timeout is hit).
 *
 * Batching of N evaluations is accomplished by maintaining a length-N array of evaluation objects,
 * and various tensors (for nnet input and output) of shape (N, ...). Each evaluate() call gets
 * assigned a particular index i with 0 <= i < N, and writes to the i'th slot of these data
 * structures. A separate evaluation thread issues the nnet evaluation and writes to the i'th slot
 * of the output data structures.
 *
 * The service has an LRU cache, which helps to avoid the costly GPU operations when possible.
 *
 * Here is a detailed description of how this implementation handles the various thread safety
 * considerations.
 *
 * There are three mutexes:
 *
 * - cache_mutex_: prevents race-conditions on cache reads/writes - especially important because
 * without locking, cache eviction can lead to a key-value pair disappearing after checking for the
 * key
 * - batch_data_.mutex: prevents race-conditions on reads/writes of batch_data_
 * - batch_metadata_.mutex: prevents race-conditions on reads/writes of batch_metadata_
 *
 * The batch_data_ member consists of:
 *
 * - input: the batch input tensor
 * - value/policy: the batch output tensors
 * - eval_ptr_data: mainly an array of N smart-pointers to a struct that has copied a slice of the
 * value/policy tensors.
 *
 * The batch_metadata_ member consists of three ints:
 *
 * - reserve_index: the next slot of batch_data_.input to write to
 * - commit_count: the number of slots of batch_data_.input that have been written to
 * - unread_count: the number of entries of batch_data_.eval_ptr_data that have not yet been read by
 * their corresponding search threads
 *
 * The loop() and evaluate() methods of NNEvaluationService have been carefully written to ensure
 * that the reads and writes of these data structures are thread-safe.
 *
 * Compiling with -DMCTS_DEBUG will enable a bunch of prints that allow you to watch the sequence of
 * operations in the interleaving threads.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
class NNEvaluationService
    : public core::CmdServerListener<core::CmdServerInteractionType::kPause>,
      public core::CmdServerListener<core::CmdServerInteractionType::kReloadWeights>,
      public core::CmdServerListener<core::CmdServerInteractionType::kMetricsRequest> {
 public:
  using GameStateTypes = core::GameStateTypes<GameState>;
  using TensorizorTypes = core::TensorizorTypes<Tensorizor>;
  using dtype = torch_util::dtype;

  using Node = mcts::Node<GameState, Tensorizor>;
  using NNEvaluation = mcts::NNEvaluation<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;

  using NNEvaluation_asptr = typename NNEvaluation::asptr;
  using NNEvaluation_sptr = typename NNEvaluation::sptr;

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

  using PolicyTransform = core::AbstractSymmetryTransform<PolicyTensor>;

  struct Request {
    Node* tree;
    search_thread_profiler_t* thread_profiler;
    int thread_id;
    core::symmetry_index_t sym_index;
  };

  struct Response {
    NNEvaluation_sptr ptr;
    bool used_cache;
  };

  /*
   * Constructs an evaluation thread and returns it.
   *
   * If another thread with the given model_filename has already been create()'d, then returns that.
   * If that returned thread does not match the thread parameters (batch_size, nn_eval_timeout_ns,
   * cache_size), then raises an exception.
   */
  static NNEvaluationService* create(const NNEvaluationServiceParams& params);

  /*
   * Instantiates the thread_ member if not yet instantiated. This spawns a new thread.
   *
   * If the thread_ member is already instantiated, then this is a no-op.
   */
  void connect();

  void disconnect();

  void set_profiling_dir(const boost::filesystem::path& profiling_dir);

  /*
   * Called by search threads. Returns immediately if we get a cache-hit. Otherwise, blocks on the
   * service thread.
   *
   * Note that historically, parallel MCTS did evaluations asynchronously. AlphaGo Zero was the
   * first version that switched to blocking evaluations.
   *
   * "Compared to the MCTS in AlphaGo Fan and AlphaGo Lee, the principal differences are...each
   * search thread simply waits for the neural network evaluation, rather than performing evaluation
   * and backup asynchronously"
   *
   * - Mastering the Game of Go without Human Knowledge (page 27)
   *
   * https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
   */
  Response evaluate(const Request&);

  void end_session();

  core::perf_stats_t get_perf_stats() override;

 private:
  using instance_map_t = std::map<boost::filesystem::path, NNEvaluationService*>;
  using cache_key_t = util::HashablePair<GameState, core::symmetry_index_t>;
  using cache_t = util::LRUCache<cache_key_t, NNEvaluation_asptr>;
  using profiler_t = nn_evaluation_service_profiler_t;

  NNEvaluationService(const NNEvaluationServiceParams& params);
  ~NNEvaluationService();

  std::string dump_key(const char* descr);

  void batch_evaluate();
  void loop();

  Response check_cache(const Request&, const cache_key_t& cache_key);
  void wait_until_batch_reservable(const Request&, std::unique_lock<std::mutex>& metadata_lock);
  int allocate_reserve_index(const Request&, std::unique_lock<std::mutex>& metadata_lock);
  void tensorize_and_transform_input(const Request& request, const cache_key_t& cache_key,
                                     int reserve_index);
  void increment_commit_count(const Request&);
  NNEvaluation_sptr get_eval(const Request&, int reserve_index,
                             std::unique_lock<std::mutex>& metadata_lock);
  void wait_until_all_read(const Request&, std::unique_lock<std::mutex>& metadata_lock);

  void wait_for_unpause();
  void reload_weights(const std::string& model_filename) override;
  void pause() override;
  void unpause() override;
  void wait_until_batch_ready();
  void wait_for_first_reservation();
  void wait_for_last_reservation();
  void wait_for_commits();

  bool active() const { return num_connections_; }

  struct eval_ptr_data_t {
    NNEvaluation_asptr eval_ptr;

    cache_key_t cache_key;
    ActionMask valid_actions;
    PolicyTransform* policy_transform;
  };

  struct tensor_group_t {
    void load_output_from(int row, torch::Tensor& torch_policy, torch::Tensor& torch_value);

    InputTensor input;
    PolicyTensor policy;
    ValueTensor value;
    core::seat_index_t current_player;
    eval_ptr_data_t eval_ptr_data;
  };

  struct batch_data_t {
    batch_data_t(int batch_size);
    ~batch_data_t();
    void copy_input_to(int num_rows, DynamicInputFloatTensor& full_input);

    std::mutex mutex;
    tensor_group_t* tensor_groups_;
  };

  static instance_map_t instance_map_;
  static int instance_count_;

  const int instance_id_;
  const NNEvaluationServiceParams params_;

  profiler_t profiler_;
  std::thread* thread_ = nullptr;
  std::mutex cache_mutex_;
  std::mutex connection_mutex_;

  std::condition_variable cv_service_loop_;
  std::condition_variable cv_evaluate_;

  core::NeuralNet net_;

  batch_data_t batch_data_;

  core::NeuralNet::input_vec_t input_vec_;
  torch::Tensor torch_input_gpu_;
  torch::Tensor torch_policy_;
  torch::Tensor torch_value_;
  DynamicInputFloatTensor full_input_;
  cache_t cache_;

  const std::chrono::nanoseconds timeout_duration_;

  time_point_t deadline_;
  struct batch_metadata_t {
    std::mutex mutex;
    int reserve_index = 0;
    int commit_count = 0;
    int unread_count = 0;
    bool accepting_reservations = true;
    std::string repr() const {
      return util::create_string("res=%d, com=%d, unr=%d, acc=%d", reserve_index, commit_count,
                                 unread_count, accepting_reservations);
    }
  };
  batch_metadata_t batch_metadata_;

  bool session_ended_ = false;
  int num_connections_ = 0;

  bool paused_ = false;
  std::mutex pause_mutex_;
  std::condition_variable cv_paused_;

  core::perf_stats_t perf_stats_;
  mutable std::mutex perf_stats_mutex_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluationService.inl>
