#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/GameServerClient.hpp"
#include "core/InputTensorizor.hpp"
#include "core/LoopControllerListener.hpp"
#include "core/NeuralNet.hpp"
#include "core/PerfStats.hpp"
#include "core/TensorTypes.hpp"
#include "core/YieldManager.hpp"
#include "search/NNEvaluation.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/NNEvaluationServiceBase.hpp"
#include "search/NNEvaluationServiceParams.hpp"
#include "search/TypeDefs.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/AllocPool.hpp"
#include "util/FiniteGroups.hpp"
#include "util/LRUCache.hpp"
#include "util/RecyclingAllocPool.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

#include <deque>
#include <memory>
#include <queue>
#include <vector>

namespace search {

/*
 * The NNEvaluationService services multiple search threads, which may belong to multiple
 * search::Manager instances (if two nnet agents are playing against each other for instance).
 *
 * The main API is the evaluate() method. It accepts an NNEvaluationRequest, which contains one or
 * more game states to evaluate, along with instructions on how to asynchronously notify a handler
 * when the evaluation is complete. The evaluation involves tensorizing the game states, adding the
 * tensors to an input batch, passing that batch to a neural network, and then writing the output
 * back to locations specified in the NNEvaluationRequest.
 *
 * Batching of N evaluations is accomplished by maintaining a length-N array of evaluation objects,
 * and various tensors (for nnet input and output) of shape (N, ...). Each evaluate() call gets
 * assigned a particular index i with 0 <= i < N, and writes to the i'th slot of these data
 * structures. A separate evaluation thread issues the nnet evaluation and writes to the i'th slot
 * of the output data structures.
 *
 * The service has an LRU cache, which helps to avoid the costly GPU operations when possible.
 *
 * Compiling with -DMCTS_NN_SERVICE_DEBUG will enable a bunch of prints that allow you to track the
 * state of the service. This is useful for debugging, but will slow down the service significantly.
 */
template <search::concepts::Traits Traits>
class NNEvaluationService
    : public NNEvaluationServiceBase<Traits>,
      public core::PerfStatsClient,
      public core::GameServerClient,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kPause>,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kReloadWeights>,
      public core::LoopControllerListener<core::LoopControllerInteractionType::kWorkerReady> {
 public:
  static constexpr const char* kCls = "NNEvaluationService";

  using sptr = std::shared_ptr<NNEvaluationService>;
  using weak_ptr = std::weak_ptr<NNEvaluationService>;

  using EvalSpec = Traits::EvalSpec;
  using Game = Traits::Game;
  using TensorTypes = core::TensorTypes<EvalSpec>;
  using InputTensorizor = core::InputTensorizor<Game>;
  using TrainingTargets = Traits::EvalSpec::TrainingTargets;

  using NeuralNet = core::NeuralNet<EvalSpec>;
  using NNEvaluation = search::NNEvaluation<Traits>;
  using NNEvaluationRequest = search::NNEvaluationRequest<Traits>;
  using NNEvaluationPool = util::AllocPool<NNEvaluation, 10, false>;

  using ActionMask = Game::Types::ActionMask;

  using InputShape = TensorTypes::InputShape;
  using InputTensor = TensorTypes::InputTensor;
  using OutputTensorTuple = TensorTypes::OutputTensorTuple;
  using OutputDataArray = TensorTypes::OutputDataArray;

  using State = Game::State;

  using RequestItem = NNEvaluationRequest::Item;

  using CacheKey = NNEvaluationRequest::CacheKey;
  using CacheKeyHasher = NNEvaluationRequest::CacheKeyHasher;

  using LRUCache = util::LRUCache<CacheKey, NNEvaluation*, CacheKeyHasher>;
  using EvalPool = util::RecyclingAllocPool<NNEvaluation>;

  class ShutDownException : public std::exception {};

  enum system_state_t : int8_t {
    kPausingScheduleLoop,
    kPausingDrainLoop,
    kPaused,

    kUnpausingScheduleLoop,
    kUnpausingDrainLoop,
    kUnpaused,

    kShuttingDownScheduleLoop,
    kShuttingDownDrainLoop,
    kShutDownComplete
  };

  NNEvaluationService(const NNEvaluationServiceParams& params, core::GameServerBase*);
  ~NNEvaluationService();

  /*
   * Constructs an evaluation service and returns it.
   *
   * If another service with the same model_filename has already been create()'d, then returns that.
   * In this case, validates that the parameters match the existing service.
   */
  static sptr create(const NNEvaluationServiceParams&, core::GameServerBase*);

  /*
   * Instantiates the thread_ member if not yet instantiated. This spawns a new thread.
   *
   * If the thread_ member is already instantiated, then this is a no-op.
   */
  void connect() override;

  void disconnect() override;

  core::yield_instruction_t evaluate(NNEvaluationRequest& request) override;

  void end_session() override;

  void update_perf_stats(core::PerfStats&) override;

  void handle_force_progress() override;

 private:
  struct TensorGroup {
    InputTensor input;
    NNEvaluation* eval;
    CacheKey cache_key;
    ActionMask valid_actions;
    group::element_t sym;
    core::action_mode_t action_mode;
    core::seat_index_t active_seat;
  };

  // BatchData represents a batch of data that is passed to the neural network for evaluation.
  // It contains a fixed-size-vector of TensorGroup objects, each of which represents a single
  // position for evaluation.
  //
  // Multiple threads can submit evaluation requests to the NNEvaluationService. The service first,
  // while holding the main_mutex_, allocates portions of BatchData's to each of the requests.
  // After allocating, it releases the main_mutex_ and allows the threads to write to their
  // allocated slots concurrently.
  struct BatchData {
    BatchData() { set_capacity(1); }
    void set_capacity(int capacity);
    void copy_input_to(int num_rows, NeuralNet&, core::pipeline_index_t);
    void load(OutputDataArray& output_data);

    template <typename Tensor>
    void load_helper(float** src, Tensor& dst);

    int capacity() const { return tensor_groups.size(); }
    void clear();
    bool frozen() const { return !accepting_allocations && write_count == allocate_count; }

    // tensor_groups is sized to capacity, and then its size never changes thereafter.
    std::vector<TensorGroup> tensor_groups;

    core::slot_context_vec_t notification_tasks;

    // TODO: I'm skeptical that sequence-id's are needed anymore, now that we are passing
    // notification task info to the evaluate() method. If so, we can remove
    // core::nn_evaluation_sequence_id_t completely, along with all the code that uses it.
    core::nn_evaluation_sequence_id_t sequence_id = 0;  // unique to each BatchData

    // Below fields are protected by the main_mutex_ member of NNEvaluationService.
    int allocate_count = 0;
    int write_count = 0;
    bool accepting_allocations = true;
  };
  using batch_data_vec_t = std::vector<BatchData*>;
  using batch_data_queue_t = std::deque<BatchData*>;

  struct BatchDataSlice {
    BatchData* batch_data;
    int start_row;
    int num_rows;
  };

  // The BatchDataSliceAllocator is responsible for allocating slices of BatchData. It must do this
  // in a thread-safe but performant manner.
  //
  // Each BatchData can be thought of as a fixed sized array of data units, like Data[B]. The
  // BatchDataSliceAllocator can be thought of as managing a sequence of BatchData's, like
  // BatchData_0, BatchData_1, BatchData_2, etc.
  //
  // The BatchDataSliceAllocator must basically support calls of the form alloc(n), responding to
  // such calls by telling the caller which BatchData to write to, and which slice of that BatchData
  // to write to. For example, it might respond to alloc(10) with BatchData_3[3:13], or maybe with
  // (BatchData_5[57:64], BatchData_6[0:3]).
  //
  // Not only must it do this allocation logic, but it must also do the work of constructing the
  // BatchData's on the fly as needed, recycling them when possible.
  //
  // The underlying implementation uses both std::atomic and mit::mutex. Usually, it can respond to
  // requests without locking the mutex. Only when the current batch_data is full does it resort to
  // locking the mutex. This is done to avoid contention on the mutex.
  class BatchDataSliceAllocator {
   public:
    BatchDataSliceAllocator(core::PerfStats& perf_stats);

    ~BatchDataSliceAllocator();

    // The main interface. The caller constructs an array BatchDataSlice[], and then passes in
    // that array to this method. The method will fill in the array with the slices that it wants to
    // allocate. The value n is the number of slots that the caller wants to allocate.
    //
    // BatchDataSliceAllocator attempts to do this without locking the mutex, relying on atomic
    // members of BatchData instead to ensure thread-safety. If it is unable to do this, then it
    // resorts to locking the passed-in mutex.
    void allocate_slices(BatchDataSlice* slices, int n, mit::mutex& main_mutex);

    void recycle(BatchData* batch_data);

    // Freezes the first BatchData in the pending list.
    void freeze_first();

    BatchData* get_first_pending_batch_data() const;
    BatchData* pop_first_pending_batch_data();

    void set_batch_size_limit(int limit);
    int batch_size_limit() const { return batch_size_limit_; }
    int pending_batch_datas_size() const { return pending_batch_datas_.size(); }

   private:
    BatchData* add_batch_data();

    int batch_size_limit_ = 1;
    core::PerfStats& perf_stats_;

    batch_data_queue_t pending_batch_datas_;
    batch_data_vec_t batch_data_reserve_;
    core::nn_evaluation_sequence_id_t next_batch_data_sequence_id_ = 1;
  };

  // Whenever we miss cache, we will need to evaluate the item later with the neural net. This
  // struct is used to store bookkeeping information to facilitate that.
  struct CacheMissInfo {
    BatchData* batch_data;  // the BatchData that we will write to
    int row;                // the row in the BatchData that we will write to
    int item_index;         // index of the item in the request
  };

  // Helper struct to pass into check_cache().
  struct CacheLookupResult {
    // Constructor accepts an array of CacheMissInfo's. The check_cache() method will populate the
    // first n entries of this array, where n is the number of cache-misses.
    CacheLookupResult(CacheMissInfo* m) : miss_infos(m) {}

    void update_notification_info(BatchData*, core::nn_evaluation_sequence_id_t id);

    CacheMissInfo* miss_infos;

    core::SearchThreadPerfStats stats;
    bool can_continue = true;
    BatchData* notifying_batch_data = nullptr;
    core::nn_evaluation_sequence_id_t notifying_sequence_id = 0;
  };

  // We empirically found that having a single LRUCache for the entire service leads to a
  // performance bottleneck, as a significant percentage of the time is spent acquiring the mutex.
  // We relieve this contention by splitting the cache into kNumHashShards shards, and using a
  // different mutex for each shard.
  struct ShardData {
    ~ShardData();
    void init(int cache_size);
    void decrement_ref_count(NNEvaluation* eval);

    mutable mit::mutex mutex;
    LRUCache eval_cache;
    EvalPool eval_pool;
  };

  // Convenience struct to sort the items in the request by hash shard.
  struct SortItem {
    auto operator<=>(const SortItem& other) const = default;
    hash_shard_t shard;
    bool fresh;
    int16_t item_index;
  };
  static_assert(sizeof(SortItem) == 4);

  struct LoadQueueItem {
    LoadQueueItem(BatchData* b = nullptr, core::pipeline_index_t p = -1)
        : batch_data(b), pipeline_index(p) {}

    BatchData* batch_data;
    core::pipeline_index_t pipeline_index;
  };
  using load_queue_t = std::queue<LoadQueueItem>;

  std::string dump_key(const char* descr);

  void activate_net();

  // For each item in the request, attempt a cache-lookup. If we get a cache-hit, set the item's
  // eval to the cached value. If we get a cache-miss, we do the following:
  //
  // 1. Create a new NNEvaluation (to be populated later)
  // 2. Insert the NNEvaluation into the cache
  // 3. Set the item's eval to that.
  // 4. Populate result.miss_infos with the cache-miss information.
  void check_cache(NNEvaluationRequest& request, CacheLookupResult& result);

  void populate_sort_items(SortItem* sort_items, NNEvaluationRequest& request);

  // return true if miss cache
  bool handle_fresh_item(NNEvaluationRequest&, CacheLookupResult&, ShardData&, int item_index);

  void write_miss_infos(NNEvaluationRequest&, CacheLookupResult&, int& miss_info_write_index,
                        int misses_for_this_shard);

  void write_to_batch(const RequestItem& item, BatchData* batch_data, int row);

  // Returns true if the notification task was registered successfully. If false, it means that
  // the batch_data was already processed by batch_evaluate() before we could register the
  // notification task.
  bool register_notification_task(const NNEvaluationRequest&, const CacheLookupResult&);

  void schedule_loop();
  void drain_loop();
  void state_loop();

  void load_initial_weights_if_necessary();
  void schedule_loop_prelude();
  void drain_loop_prelude();

  BatchData* get_next_batch_data(core::NNEvalScheduleLoopPerfStats&);
  void schedule_batch(BatchData* batch_data, core::NNEvalScheduleLoopPerfStats&);
  bool load_queue_item(LoadQueueItem&);  // return true if item was loaded
  void drain_batch(const LoadQueueItem&);

  void reload_weights(const std::vector<char>& buf) override;
  void pause() override;
  void unpause() override;

  static inline int instance_count_ = 0;

  const int instance_id_;
  const NNEvaluationServiceParams params_;
  const int num_game_threads_ = 0;

  mit::thread schedule_thread_;
  mit::thread drain_thread_;
  mit::thread state_thread_;

  mutable mit::mutex main_mutex_;
  mutable mit::mutex perf_stats_mutex_;
  mit::condition_variable cv_main_;

  NeuralNet net_;

  ShardData shard_datas_[kNumHashShards];

  bool session_ended_ = false;
  int num_connections_ = 0;

  core::nn_evaluation_sequence_id_t last_evaluated_sequence_id_ = 0;

  bool initial_weights_loaded_ = false;
  bool ready_ = false;
  bool in_schedule_loop_prelude_ = false;
  bool in_drain_loop_prelude_ = false;
  system_state_t system_state_ = kUnpaused;

  core::PerfStats perf_stats_;
  BatchDataSliceAllocator batch_data_slice_allocator_;
  load_queue_t load_queue_;
  core::YieldManager* yield_manager_ = nullptr;
  core::GameServerBase* server_ = nullptr;
};

}  // namespace search

#include "inline/search/NNEvaluationService.inl"
