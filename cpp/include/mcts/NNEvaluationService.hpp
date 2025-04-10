#pragma once

#include <core/BasicTypes.hpp>
#include <core/LoopControllerClient.hpp>
#include <core/LoopControllerListener.hpp>
#include <core/concepts/Game.hpp>
#include <core/NeuralNet.hpp>
#include <core/PerfStats.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/NNEvaluationRequest.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>
#include <mcts/NNEvaluationServiceParams.hpp>
#include <mcts/Node.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/AllocPool.hpp>
#include <util/FiniteGroups.hpp>
#include <util/TorchUtil.hpp>

#include <condition_variable>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
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
template <core::concepts::Game Game>
class NNEvaluationService
    : public NNEvaluationServiceBase<Game>,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kPause>,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kReloadWeights>,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kMetricsRequest> {
 public:
  using Node = mcts::Node<Game>;
  using NNEvaluation = mcts::NNEvaluation<Game>;
  using NNEvaluationRequest = mcts::NNEvaluationRequest<Game>;
  using NNEvaluationPool = util::AllocPool<NNEvaluation, 10, false>;

  using ActionMask = Game::Types::ActionMask;
  using InputTensor = Game::InputTensorizor::Tensor;
  using PolicyTensor = Game::Types::PolicyTensor;
  using ValueTensor = Game::TrainingTargets::ValueTarget::Tensor;
  using ActionValueTensor = Game::Types::ActionValueTensor;

  using InputShape = InputTensor::Dimensions;
  using PolicyShape = Game::Types::PolicyShape;
  using ValueShape = ValueTensor::Dimensions;
  using ActionValueShape = Game::Types::ActionValueShape;

  using DynamicInputTensor = Eigen::Tensor<float, InputShape::count + 1, Eigen::RowMajor>;

  using State = Game::State;
  using InputTensorizor = Game::InputTensorizor;

  using RequestItem = NNEvaluationRequest::Item;
  using instance_map_t = std::map<std::string, NNEvaluationService*>;

  using cache_key_t = NNEvaluationRequest::cache_key_t;
  using profiler_t = nn_evaluation_service_profiler_t;

  // EvalCache is essentially a LRU hash map. A few details that warrant defining a specialized
  // data structure rather than using a generic one:
  //
  // 1. In order to make the nn eval service thread as lean as possible, we want all hashing to be
  //    done in other threads. This means that some methods accept a hash as an argument, rather
  //    than computing it internally.
  //
  // 2. Eviction from the cache decrements the ref count of the NNEvaluation object. When the
  //    ref count reaches zero, we recycle the NNEvaluation object, rather than delete'ing it.
  //
  // Thread-safety is guaranteed by the cache_mutex_ member of NNEvaluationService, so we don't
  // require a mutex internally.
  class EvalCache {
   public:
    using hash_t = uint64_t;
    using value_creation_func_t = std::function<NNEvaluation*()>;
    using eviction_func_t = std::function<void(NNEvaluation*)>;

    struct Entry {
      Entry(const cache_key_t& k, hash_t h, NNEvaluation* e) : key(k), hash(h), eval(e) {}

      cache_key_t key;
      hash_t hash;
      NNEvaluation* eval;
    };

    using Key = hash_t;
    using EntryList = std::list<Entry>;
    using EntryListIterator = EntryList::iterator;
    using MapValue = std::list<EntryListIterator>;
    using Map = std::unordered_map<uint64_t, MapValue>;

    EvalCache(eviction_func_t f, size_t capacity) : eviction_handler_(f), capacity_(capacity) {}

    // If key is already in the cache, returns the value for it. Otherwise, adds a mapping, calling
    // value_creator() to create the value. If the cache is full, evicts the least recently used
    // item, calling eviction_handler() on it.
    //
    // hash is the hash of the key.
    NNEvaluation* insert(const cache_key_t& key, hash_t hash, value_creation_func_t value_creator);

    void clear();

   private:
    size_t size() const { return size_; }

    void evict(MapValue* protected_list);

    const eviction_func_t eviction_handler_;
    const size_t capacity_;

    // list_: [(cache_key, hash, eval), ...] in most-recently-used order
    // map_: {hash -> [iter...]}, where each iter is an iterator into list_
    EntryList list_;
    Map map_;
    size_t size_ = 0;
  };

  /*
   * Constructs an evaluation service and returns it.
   *
   * If another service with the same model_filename has already been create()'d, then returns that.
   * In this case, validates that the parameters match the existing service.
   */
  static NNEvaluationService* create(const NNEvaluationServiceParams& params);

  /*
   * Instantiates the thread_ member if not yet instantiated. This spawns a new thread.
   *
   * If the thread_ member is already instantiated, then this is a no-op.
   */
  void connect() override;

  void disconnect() override;

  NNEvaluationResponse evaluate(NNEvaluationRequest& request) override;
  void wait_for(core::nn_evaluation_sequence_id_t sequence_id) override;

  void end_session() override;

  core::PerfStats get_perf_stats() override;

 private:
  struct CacheLookupResult {
    int size() const { return non_pending_hits + pending_hits + misses; }

    int non_pending_hits = 0;  // item in cache and in non-pending state
    int pending_hits = 0;      // item in cache and in pending state
    int misses = 0;            // item not in cache
    core::nn_evaluation_sequence_id_t max_pending_sequence_id = 0;
  };

  struct TensorGroup {
    void load_output_from(int row, torch::Tensor& torch_policy, torch::Tensor& torch_value,
                          torch::Tensor& torch_action_value);

    InputTensor input;
    PolicyTensor policy;
    ValueTensor value;
    ActionValueTensor action_values;

    NNEvaluation* eval;
    cache_key_t cache_key;
    ActionMask valid_actions;
    group::element_t sym;
    core::action_mode_t action_mode;
    core::seat_index_t active_seat;
  };

  struct BatchData {
    BatchData(int capacity);
    void copy_input_to(int num_rows, DynamicInputTensor& full_input);
    int capacity() const { return tensor_groups.size(); }
    void clear();
    bool frozen() const { return !accepting_allocations && write_count == allocate_count; }

    // tensor_groups is sized to capacity, and then its size never changes thereafter.
    std::vector<TensorGroup> tensor_groups;

    core::nn_evaluation_sequence_id_t sequence_id = 0;

    int allocate_count = 0;  // protected by NNEvaluationService::batch_data_mutex_
    int write_count = 0;
    bool accepting_allocations = true;
  };
  using batch_data_vec_t = std::vector<BatchData*>;

  struct BatchSlab {
    BatchData* batch_data;
    int offset;
    int size;
  };

  NNEvaluationService(const NNEvaluationServiceParams& params);
  ~NNEvaluationService();

  std::string dump_key(const char* descr);

  void set_profiling_dir(const boost::filesystem::path& profiling_dir);

  // For each item in the request, check if the cache has a value for it. If so, set the item to
  // that value. Else, insert a placeholder in the cache.
  CacheLookupResult check_cache(NNEvaluationRequest&);
  void decrement_ref_count(NNEvaluation* eval);

  void allocate_slabs(BatchSlab* slabs, int n, int limit);
  BatchData* add_batch_data();  // assumes batch_data_mutex_ is held
  void write_to_batch(const RequestItem& item, BatchData* batch_data, int row);
  void update_perf_stats(const CacheLookupResult& result);

  void loop();
  void set_deadline();
  void load_initial_weights_if_necessary();
  void wait_for_unpause();
  void wait_until_batch_ready();
  void batch_evaluate();

  void reload_weights(const std::vector<char>& buf, const std::string& cuda_device) override;
  void pause() override;
  void unpause() override;
  bool active() const { return num_connections_; }

  static instance_map_t instance_map_;
  static int instance_count_;

  const int instance_id_;
  const NNEvaluationServiceParams params_;

  profiler_t profiler_;
  std::thread* thread_ = nullptr;

  mutable std::mutex cache_mutex_;
  mutable std::mutex connection_mutex_;
  mutable std::mutex net_weights_mutex_;
  mutable std::mutex main_mutex_;
  mutable std::mutex perf_stats_mutex_;

  std::condition_variable cv_net_weights_;
  std::condition_variable cv_main_;
  std::condition_variable cv_eval_;

  core::NeuralNet net_;

  core::NeuralNet::input_vec_t input_vec_;
  torch::Tensor torch_input_gpu_;
  torch::Tensor torch_policy_;
  torch::Tensor torch_value_;
  torch::Tensor torch_action_value_;
  DynamicInputTensor full_input_;

  EvalCache eval_cache_;

  const std::chrono::nanoseconds timeout_duration_;

  time_point_t deadline_;

  batch_data_vec_t pending_batch_datas_;
  batch_data_vec_t batch_data_reserve_;
  core::nn_evaluation_sequence_id_t next_batch_data_sequence_id_ = 1;

  bool session_ended_ = false;
  int num_connections_ = 0;

  core::nn_evaluation_sequence_id_t last_evaluated_sequence_id_ = 0;

  bool initial_weights_loaded_ = false;
  bool ready_ = false;
  bool skip_next_pause_receipt_ = false;
  bool paused_ = false;

  core::PerfStats perf_stats_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluationService.inl>
