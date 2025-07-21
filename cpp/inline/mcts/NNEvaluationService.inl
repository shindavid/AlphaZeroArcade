#include "mcts/NNEvaluationService.hpp"
#include "util/Asserts.hpp"
#include "util/KeyValueDumper.hpp"
#include "util/LoggingUtil.hpp"

#include <boost/json/src.hpp>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

#include <cstdint>
#include <spanstream>

namespace mcts {

template <core::concepts::Game Game>
int NNEvaluationService<Game>::instance_count_ = 0;

template <core::concepts::Game Game>
NNEvaluationService<Game>* NNEvaluationService<Game>::create(
  const NNEvaluationServiceParams& params, core::GameServerBase* server) {
  static instance_map_t instance_map;

  auto it = instance_map.find(params.model_filename);
  if (it == instance_map.end()) {
    auto instance = new NNEvaluationService(params, server);

    instance_map[params.model_filename] = instance;
    if (instance_map.size() > 1) {
      server->handle_alternating_mode_recommendation();
    }
    return instance;
  }
  NNEvaluationService* instance = it->second;
  if (instance->params_ != params) {
    throw util::CleanException("Conflicting NNEvaluationService::create() calls");
  }
  return instance;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::connect() {
  mit::unique_lock lock(main_mutex_);
  bool first_connect = (num_connections_ == 0);
  num_connections_++;

  if (first_connect) {
    schedule_thread_ = mit::thread([&] { this->schedule_loop(); });
    drain_thread_ = mit::thread([&] { this->drain_loop(); });
    state_thread_ = mit::thread([&] { this->state_loop(); });
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::disconnect() {
  mit::unique_lock lock(main_mutex_);
  num_connections_--;
  lock.unlock();
  cv_main_.notify_all();

  for (auto* thread : {&schedule_thread_, &drain_thread_, &state_thread_}) {
    if (thread->joinable()) {
      thread->join();
    }
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::CacheLookupResult::update_notification_info(
  BatchData* batch_data, core::nn_evaluation_sequence_id_t id) {
  if (id > notifying_sequence_id) {
    notifying_batch_data = batch_data;
    notifying_sequence_id = id;
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::ShardData::init(int cache_size) {
  eval_cache.set_capacity(cache_size);
  eval_cache.set_eviction_handler([&](NNEvaluation* e) { decrement_ref_count(e); });
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::ShardData::decrement_ref_count(NNEvaluation* eval) {
  if (eval->decrement_ref_count()) {
    eval_pool.free(eval);
  }
}

template <core::concepts::Game Game>
inline NNEvaluationService<Game>::NNEvaluationService(const NNEvaluationServiceParams& params,
                                                      core::GameServerBase* server)
    : core::PerfStatsClient(),
      core::GameServerClient(server),
      instance_id_(instance_count_++),
      params_(params),
      num_game_threads_(server->num_game_threads()),
      net_(cuda_util::cuda_device_to_ordinal(params_.cuda_device)),
      batch_data_slice_allocator_(perf_stats_),
      server_(server) {
  if (!params.model_filename.empty()) {
    net_.load_weights(params.model_filename.c_str());
    activate_net();
  }

  for (int i = 0; i < kNumHashShards; i++) {
    shard_datas_[i].init(params_.cache_size / kNumHashShards);
  }

  initial_weights_loaded_ = net_.loaded();
  if (core::LoopControllerClient::initialized()) {
    core::LoopControllerClient::get()->add_listener(this);
  } else {
    if (!net_.loaded()) {
      throw util::CleanException(
        "MCTS player configured without --model-filename/-m and without "
        "--no-model, but --loop-controller-* options not specified");
    }
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchData::set_capacity(int capacity) {
  tensor_groups.resize(capacity);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchData::copy_input_to(int num_rows, NeuralNet& net,
                                                         core::pipeline_index_t pipeline_index) {
  float* input_ptr = net.get_input_ptr(pipeline_index);
  constexpr size_t input_size = InputShape::total_size;
  int r = 0;
  for (int row = 0; row < num_rows; row++) {
    const TensorGroup& group = tensor_groups[row];
    memcpy(input_ptr + r, group.input.data(), input_size * sizeof(float));
    r += input_size;
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchData::load(const float* policy_batch_data,
                                                const float* value_batch_data,
                                                const float* action_values_batch_data) {
  const float* policy_data = policy_batch_data;
  const float* value_data = value_batch_data;
  const float* action_values_data = action_values_batch_data;

  for (int i = 0; i < write_count; ++i) {
    TensorGroup& group = tensor_groups[i];

    PolicyTensor policy;
    ValueTensor value;
    ActionValueTensor action_values;

    std::copy_n(policy_data, policy.size(), policy.data());
    std::copy_n(value_data, value.size(), value.data());
    std::copy_n(action_values_data, action_values.size(), action_values.data());

    policy_data += PolicyShape::total_size;
    value_data += ValueShape::total_size;
    action_values_data += ActionValueShape::total_size;

    // WARNING: this function all modifies policy/value/action_values in-place. So we should be
    // careful not to read them after this call.
    group.eval->init(policy, value, action_values, group.valid_actions, group.sym,
                     group.active_seat, group.action_mode);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchData::clear() {
  sequence_id = 0;
  allocate_count = 0;
  write_count = 0;
  accepting_allocations = true;
  notification_tasks.clear();
}

template <core::concepts::Game Game>
NNEvaluationService<Game>::BatchDataSliceAllocator::BatchDataSliceAllocator(
  core::PerfStats& perf_stats)
    : perf_stats_(perf_stats) {
  add_batch_data();
}

template <core::concepts::Game Game>
NNEvaluationService<Game>::BatchDataSliceAllocator::~BatchDataSliceAllocator() {
  while (!pending_batch_datas_.empty()) {
    BatchData* batch_data = pending_batch_datas_.front();
    pending_batch_datas_.pop_front();
    delete batch_data;
  }

  for (BatchData* batch_data : batch_data_reserve_) {
    delete batch_data;
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchDataSliceAllocator::allocate_slices(BatchDataSlice* slices,
                                                                         int n,
                                                                         mit::mutex& main_mutex) {
  mit::unique_lock lock(main_mutex);

  int slice_index = 0;
  BatchData* batch_data = pending_batch_datas_.back();
  while (n) {
    BatchDataSlice& slice = slices[slice_index++];

    int m = std::min(n, batch_data->capacity() - batch_data->allocate_count);
    RELEASE_ASSERT(m > 0);
    n -= m;

    slice.batch_data = batch_data;
    slice.start_row = batch_data->allocate_count;
    slice.num_rows = m;
    batch_data->allocate_count += m;

    if (batch_data->allocate_count == batch_data->capacity()) {
      batch_data->accepting_allocations = false;
      batch_data = add_batch_data();
    }
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchDataSliceAllocator::recycle(BatchData* batch_data) {
  LOG_DEBUG("<-- NNEvaluationService: Recycling batch data {}", batch_data->sequence_id);
  batch_data->clear();
  batch_data_reserve_.push_back(batch_data);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchDataSliceAllocator::freeze_first() {
  if (pending_batch_datas_.empty()) return;
  BatchData* batch_data = pending_batch_datas_.front();
  if (batch_data->allocate_count == 0) return;
  if (!batch_data->accepting_allocations) return;

  LOG_DEBUG("<-- NNEvaluationService: Freezing batch data {} (alloc:{} write:{})",
            batch_data->sequence_id, batch_data->allocate_count, batch_data->write_count);

  batch_data->accepting_allocations = false;
  if (pending_batch_datas_.size() == 1) {
    add_batch_data();
  }
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::BatchData*
NNEvaluationService<Game>::BatchDataSliceAllocator::get_first_pending_batch_data() const {
  if (!pending_batch_datas_.empty()) {
    return pending_batch_datas_.front();
  }
  return nullptr;
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::BatchData*
NNEvaluationService<Game>::BatchDataSliceAllocator::pop_first_pending_batch_data() {
  if (!pending_batch_datas_.empty()) {
    BatchData* batch_data = pending_batch_datas_.front();
    pending_batch_datas_.pop_front();
    if (pending_batch_datas_.empty()) {
      add_batch_data();
    }
    return batch_data;
  }
  return nullptr;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchDataSliceAllocator::set_batch_size_limit(int limit) {
  batch_size_limit_ = limit;
  for (auto& batch_data : batch_data_reserve_) {
    batch_data->set_capacity(limit);
  }
  for (auto batch_data : pending_batch_datas_) {
    batch_data->set_capacity(limit);
  }
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::BatchData*
NNEvaluationService<Game>::BatchDataSliceAllocator::add_batch_data() {
  // Assumes mutex_ is locked
  BatchData* batch_data;
  if (batch_data_reserve_.empty()) {
    batch_data = new BatchData();
    batch_data->set_capacity(batch_size_limit_);
    perf_stats_.nn_eval_schedule_loop_stats.batch_datas_allocated++;
  } else {
    batch_data = batch_data_reserve_.back();
    batch_data_reserve_.pop_back();
  }
  batch_data->sequence_id = next_batch_data_sequence_id_++;
  pending_batch_datas_.push_back(batch_data);
  return batch_data;
}

template <core::concepts::Game Game>
NNEvaluationService<Game>::~NNEvaluationService() {
  disconnect();
}

template <core::concepts::Game Game>
core::yield_instruction_t NNEvaluationService<Game>::evaluate(NNEvaluationRequest& request) {
  if (request.num_fresh_items() == 0) {
    return core::kContinue;
  }

  int n = request.num_fresh_items();
  CacheMissInfo miss_infos[n];
  CacheLookupResult result(miss_infos);
  check_cache(request, result);

  core::yield_instruction_t yield_instruction = core::kContinue;
  if (!result.can_continue) {
    // Write to batches
    core::PerfClocker clocker(result.stats.batch_write_time_ns);
    if (result.stats.cache_misses) {
      CacheMissInfo& last_miss_info = miss_infos[result.stats.cache_misses - 1];
      BatchData* batch_data = last_miss_info.batch_data;
      result.update_notification_info(batch_data, batch_data->sequence_id);
    }

    // notifying_sequence_id is either set above because cache_misses > 0, or it is set inside
    // handle_fresh_item() in the hit_cache && eval->pending() case. There is no other way that
    // can_continue can be set to false.
    //
    // TODO: downgrade some of these RELEASE_ASSERT's to DEBUG_ASSERT's.
    RELEASE_ASSERT(result.notifying_sequence_id > 0);
    if (!register_notification_task(request, result)) {
      // This means that we hit a race condition where the corresponding batch data was evaluated
      // before we could register the notification task. If we blindly added to the notification
      // task without checking for this, the GameServer would end up waiting for a notification that
      // would never arrive.

      RELEASE_ASSERT(result.stats.cache_misses == 0, "Unexpected cache_misses={}",
                     result.stats.cache_misses);
      for (auto& item : request.fresh_items()) {
        RELEASE_ASSERT(!item.eval()->pending(), "Unexpected pending item");
      }

      yield_instruction = core::kContinue;  // not needed but just here for clarity
    } else {
      for (int i = 0; i < result.stats.cache_misses; ++i) {
        CacheMissInfo& miss_info = miss_infos[i];
        RequestItem& item = request.get_fresh_item(miss_info.item_index);
        BatchData* batch_data = miss_info.batch_data;
        int row = miss_info.row;
        write_to_batch(item, batch_data, row);
      }

      mit::unique_lock lock(main_mutex_);
      for (int i = 0; i < result.stats.cache_misses; ++i) {
        CacheMissInfo& miss_info = miss_infos[i];
        BatchData* batch_data = miss_info.batch_data;
        batch_data->write_count++;
      }
      lock.unlock();

      cv_main_.notify_all();
      yield_instruction = core::kYield;
    }
  }

  mit::unique_lock perf_stats_lock(perf_stats_mutex_);
  perf_stats_.update(result.stats);
  perf_stats_lock.unlock();

  return yield_instruction;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::end_session() {
  if (session_ended_) return;

  core::PerfStats stats = core::PerfStatsRegistry::instance()->get_perf_stats();
  stats.calibrate(this->num_game_threads_);

  int64_t cache_hits = stats.search_thread_stats.cache_hits;
  int64_t cache_misses = stats.search_thread_stats.cache_misses;
  int64_t cache_attempts = cache_hits + cache_misses;
  float cache_hit_pct = cache_attempts > 0 ? 100.0 * cache_hits / cache_attempts : 0.0f;

  int64_t positions_evaluated = stats.nn_eval_schedule_loop_stats.positions_evaluated;
  int64_t batches_evaluated = stats.nn_eval_schedule_loop_stats.batches_evaluated;
  int64_t full_batches_evaluated = stats.nn_eval_schedule_loop_stats.full_batches_evaluated;

  int64_t wait_for_game_slot_time_ns = stats.search_thread_stats.wait_for_game_slot_time_ns;
  int64_t cache_mutex_acquire_time_ns = stats.search_thread_stats.cache_mutex_acquire_time_ns;
  int64_t cache_insert_time_ns = stats.search_thread_stats.cache_insert_time_ns;
  int64_t batch_prepare_time_ns = stats.search_thread_stats.batch_prepare_time_ns;
  int64_t batch_write_time_ns = stats.search_thread_stats.batch_write_time_ns;
  int64_t wait_for_nn_eval_time_ns = stats.search_thread_stats.wait_for_nn_eval_time_ns;
  int64_t mcts_time_ns = stats.search_thread_stats.mcts_time_ns;

  int64_t wait_for_search_threads_time_ns =
    stats.nn_eval_schedule_loop_stats.wait_for_search_threads_time_ns;
  int64_t pipeline_wait_time_ns = stats.nn_eval_schedule_loop_stats.pipeline_wait_time_ns;
  int64_t pipeline_schedule_time_ns = stats.nn_eval_schedule_loop_stats.pipeline_schedule_time_ns;

  int batch_datas_allocated = stats.nn_eval_schedule_loop_stats.batch_datas_allocated;

  float max_batch_pct = 100.0 * full_batches_evaluated / std::max(int64_t(1), batches_evaluated);
  float avg_batch_size = 1.0 * positions_evaluated / std::max(int64_t(1), batches_evaluated);

  float wait_for_game_slot_time_s = 1e-9 * wait_for_game_slot_time_ns;
  float cache_mutex_acquire_time_s = 1e-9 * cache_mutex_acquire_time_ns;
  float cache_insert_time_s = 1e-9 * cache_insert_time_ns;
  float batch_prepare_time_s = 1e-9 * batch_prepare_time_ns;
  float batch_write_time_s = 1e-9 * batch_write_time_ns;
  float wait_for_nn_eval_time_s = 1e-9 * wait_for_nn_eval_time_ns;
  float mcts_time_s = 1e-9 * mcts_time_ns;

  float per_batch_wait_for_search_threads_time_ms =
    1e-6 * wait_for_search_threads_time_ns / batches_evaluated;
  float per_batch_pipeline_wait_time_ms = 1e-6 * pipeline_wait_time_ns / batches_evaluated;
  float per_batch_pipeline_schedule_time_ms = 1e-6 * pipeline_schedule_time_ns / batches_evaluated;

  auto dump = [&](const char* key, const char* fmt, auto value) {
    util::KeyValueDumper::add(dump_key(key), fmt, value);
  };

  dump("cache hits", "%ld", cache_hits);
  dump("cache misses", "%ld", cache_misses);
  dump("cache hit rate", "%.2f%%", cache_hit_pct);
  dump("evaluated positions", "%ld", positions_evaluated);
  dump("batches evaluated", "%ld", batches_evaluated);
  dump("max batch pct", "%.2f%%", max_batch_pct);
  dump("avg batch size", "%.2f", avg_batch_size);
  dump("batch datas allocated", "%d", batch_datas_allocated);
  dump("search-thread total wait for game slot time", "%.3fs", wait_for_game_slot_time_s);
  dump("search-thread total cache mutex acquire time", "%.3fs", cache_mutex_acquire_time_s);
  dump("search-thread total cache insert time", "%.3fs", cache_insert_time_s);
  dump("search-thread total batch prepare time", "%.3fs", batch_prepare_time_s);
  dump("search-thread total batch write time", "%.3fs", batch_write_time_s);
  dump("search-thread total wait for nn eval time", "%.3fs", wait_for_nn_eval_time_s);
  dump("search-thread total mcts time", "%.3fs", mcts_time_s);
  dump("nn-eval per-batch wait for search threads time", "%.3fms",
       per_batch_wait_for_search_threads_time_ms);
  dump("nn-eval per-batch pipeline wait time", "%.3fms", per_batch_pipeline_wait_time_ms);
  dump("nn-eval per-batch pipeline schedule time", "%.3fms", per_batch_pipeline_schedule_time_ms);
  session_ended_ = true;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::update_perf_stats(core::PerfStats& perf_stats) {
  mit::unique_lock lock(perf_stats_mutex_);
  core::PerfStats perf_stats_copy = perf_stats_;
  new (&perf_stats_) core::PerfStats();
  lock.unlock();

  perf_stats += perf_stats_copy;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::handle_force_progress() {
  mit::unique_lock lock(main_mutex_);
  LOG_DEBUG("<-- {}::{}() size={}", kCls, __func__,
            batch_data_slice_allocator_.pending_batch_datas_size());

  batch_data_slice_allocator_.freeze_first();
  lock.unlock();
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
std::string NNEvaluationService<Game>::dump_key(const char* descr) {
  return std::format("NN-{} {}", instance_id_, descr);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::activate_net() {
  if (net_.activated()) return;

  net_.activate(params_.num_pipelines);
  batch_data_slice_allocator_.set_batch_size_limit(net_.batch_size());

  LOG_DEBUG("NNEvaluationService: activated NeuralNet with {} pipelines (batch-size: {})",
            params_.num_pipelines, net_.batch_size());
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::check_cache(NNEvaluationRequest& request,
                                            CacheLookupResult& result) {
  int m = request.num_stale_items();
  int n = request.num_fresh_items();
  int s = m + n;
  DEBUG_ASSERT(n > 0);

  // Sort the items by hash shard, so that we acquire each shard mutex at most once when iterating.
  SortItem sort_items[m + n];
  populate_sort_items(sort_items, request);

  int miss_info_write_index = 0;
  int misses_for_this_shard = 0;
  for (int i = 0; i < s; ++i) {
    SortItem& sort_item = sort_items[i];
    ShardData& shard = shard_datas_[sort_item.shard];
    bool fresh = sort_item.fresh;
    int item_index = sort_item.item_index;
    bool new_shard = (i == 0 || sort_items[i].shard != sort_items[i - 1].shard);

    if (new_shard) {
      core::PerfClocker clocker(result.stats.cache_mutex_acquire_time_ns);
      shard.mutex.lock();  // Lock can be held across loop iterations, thanks to sorting
    }

    if (!fresh) {
      // Lazily decrement old reference counts. We do this here to ensure that all reference-count
      // changes to all NNEvaluation objects are done under the same lock. If we did not do it like
      // this, then we would need to use various synchronization primitives like
      // std::atomic<std::shared_ptr> for thread-safety, which would add significant overhead.
      RequestItem& item = request.get_stale_item(item_index);
      shard.decrement_ref_count(item.eval());
    } else {
      if (handle_fresh_item(request, result, shard, item_index)) {
        misses_for_this_shard++;
      }
    }

    bool last_in_shard = (i == s - 1 || sort_items[i].shard != sort_items[i + 1].shard);
    if (last_in_shard) {
      if (misses_for_this_shard) {
        write_miss_infos(request, result, miss_info_write_index, misses_for_this_shard);
        misses_for_this_shard = 0;
      }

      shard.mutex.unlock();
    }
  }

  request.clear_stale_items();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::populate_sort_items(SortItem* sort_items,
                                                    NNEvaluationRequest& request) {
  int m = request.num_stale_items();
  int n = request.num_fresh_items();
  DEBUG_ASSERT(n > 0);

  // First, instantiate the SortItem's.
  int s = 0;
  for (int i = 0; i < m; ++i) {
    sort_items[s].shard = request.get_stale_item(i).hash_shard();
    sort_items[s].fresh = false;
    sort_items[s].item_index = i;
    s++;
  }
  for (int i = 0; i < n; ++i) {
    sort_items[s].shard = request.get_fresh_item(i).hash_shard();
    sort_items[s].fresh = true;
    sort_items[s].item_index = i;
    s++;
  }

  DEBUG_ASSERT(s == m + n);

  // Now use std::sort() to group by hash shard.
  std::sort(sort_items, sort_items + s);
}

template <core::concepts::Game Game>
bool NNEvaluationService<Game>::handle_fresh_item(NNEvaluationRequest& request,
                                                  CacheLookupResult& result, ShardData& shard,
                                                  int item_index) {
  core::PerfClocker clocker(result.stats.cache_insert_time_ns);
  RequestItem& item = request.get_fresh_item(item_index);
  RELEASE_ASSERT(item.eval() == nullptr);

  bool hit_cache = true;
  const CacheKey& cache_key = item.cache_key();

  auto value_creator = [&]() {
    hit_cache = false;
    auto eval = shard.eval_pool.alloc();
    eval->clear();
    return eval;
  };

  NNEvaluation* eval = shard.eval_cache.insert_if_missing(cache_key, value_creator);
  eval->increment_ref_count();  // ref_count++ because item is now holding it
  item.set_eval(eval);
  if (hit_cache) {
    result.stats.cache_hits++;
    if (eval->pending()) {
      BatchData* eval_data = eval->template get_aux<BatchData>();
      result.update_notification_info(eval_data, eval->eval_sequence_id());
      result.can_continue = false;
    }
  } else {
    eval->increment_ref_count();  // ref_count++ because cache is now holding it
    result.miss_infos[result.stats.cache_misses].item_index = item_index;
    result.stats.cache_misses++;
    result.can_continue = false;
    return true;
  }
  return false;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::write_miss_infos(NNEvaluationRequest& request,
                                                 CacheLookupResult& result,
                                                 int& miss_info_write_index,
                                                 int misses_for_this_shard) {
  core::PerfClocker clocker(result.stats.batch_prepare_time_ns);

  // Assign the missed items to BatchData's.
  //
  // Don't actually write to the BatchData's yet, since we don't need to do that under the
  // shard's mutex lock.

  int B = batch_data_slice_allocator_.batch_size_limit();
  int M = misses_for_this_shard;
  int max_slices_needed = (B + M - 2) / B + 1;  // 1 + ceil((M-1)/B)

  BatchDataSlice slices[max_slices_needed];
  batch_data_slice_allocator_.allocate_slices(slices, M, main_mutex_);

  int slice_index = 0;
  int slice_offset = 0;
  for (int j = 0; j < misses_for_this_shard; ++j) {
    CacheMissInfo& miss_info = result.miss_infos[miss_info_write_index++];
    RequestItem& item = request.get_fresh_item(miss_info.item_index);

    BatchDataSlice& slice = slices[slice_index];
    BatchData* batch_data = slice.batch_data;
    miss_info.batch_data = batch_data;
    miss_info.row = slice.start_row + slice_offset;

    item.eval()->set_aux(batch_data);
    item.eval()->set_eval_sequence_id(batch_data->sequence_id);
    slice_offset++;
    if (slice_offset == slices[slice_index].num_rows) {
      slice_index++;
      slice_offset = 0;
    }
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::write_to_batch(const RequestItem& item, BatchData* batch_data,
                                               int row) {
  const CacheKey& cache_key = item.cache_key();

  const auto& stable_data = item.node()->stable_data();
  const ActionMask& valid_action_mask = stable_data.valid_action_mask;
  core::seat_index_t active_seat = stable_data.active_seat;
  core::action_mode_t action_mode = stable_data.action_mode;
  group::element_t sym = item.sym();
  group::element_t inverse_sym = Game::SymmetryGroup::inverse(sym);

  auto input = item.compute_over_history([&](auto begin, auto end) {
    for (auto pos = begin; pos != end; pos++) {
      Game::Symmetries::apply(*pos, sym);
    }
    auto out = InputTensorizor::tensorize(begin, end - 1);
    for (auto pos = begin; pos != end; pos++) {
      Game::Symmetries::apply(*pos, inverse_sym);
    }
    return out;
  });

  TensorGroup& group = batch_data->tensor_groups[row];
  group.input = input;
  group.eval = item.eval();
  group.cache_key = cache_key;
  group.valid_actions = valid_action_mask;
  group.sym = sym;
  group.action_mode = action_mode;
  group.active_seat = active_seat;
}

template <core::concepts::Game Game>
bool NNEvaluationService<Game>::register_notification_task(const NNEvaluationRequest& request,
                                                           const CacheLookupResult& result) {
  // NOTE: in principle, we can initialize yield_manager_ at startup to avoid doing it here.
  // There should only ever be one yield_manager_ for the entire process. We do it here to
  // avoid having to pass it around all over the place during initialization.
  const core::YieldNotificationUnit& unit = request.notification_unit();
  if (this->yield_manager_ == nullptr) {
    yield_manager_ = unit.yield_manager;
  } else {
    // Verify that there is only one
    DEBUG_ASSERT(yield_manager_ == unit.yield_manager);
  }

  BatchData* batch_data = result.notifying_batch_data;
  core::nn_evaluation_sequence_id_t seq = result.notifying_sequence_id;

  LOG_DEBUG("<-- {}::{}() acquiring mutex...", kCls, __func__);
  mit::unique_lock lock(main_mutex_);
  if (last_evaluated_sequence_id_ < seq) {
    LOG_DEBUG("<!-- {}::{} REJECT last={} seq={} slot={}:{}", kCls, __func__,
              last_evaluated_sequence_id_, seq, unit.slot_context().slot,
              unit.slot_context().context);

    RELEASE_ASSERT(batch_data, "null batch_data");
    RELEASE_ASSERT(batch_data->sequence_id == seq, "batch_data->sequence_id:{} != seq:{}",
                   batch_data->sequence_id, seq);
    batch_data->notification_tasks.push_back(unit.slot_context());
    return true;
  } else {
    LOG_DEBUG("<!-- {}::{} ACCEPT seq={} slot={}:{}", kCls, __func__, seq, unit.slot_context().slot,
              unit.slot_context().context);

    return false;
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::schedule_loop() {
  try {
    while (system_state_ != kShuttingDownScheduleLoop) {
      core::NNEvalScheduleLoopPerfStats schedule_loop_stats;

      load_initial_weights_if_necessary();
      schedule_loop_prelude();
      activate_net();
      BatchData* batch_data = get_next_batch_data(schedule_loop_stats);
      schedule_batch(batch_data, schedule_loop_stats);
    }
  } catch (const ShutDownException&) {
    // This is expected when the state loop is shutting down.
    LOG_DEBUG("{}::{}() caught ShutDownException, exiting...", kCls, __func__);
  }

  system_state_ = kShuttingDownDrainLoop;
  in_schedule_loop_prelude_ = false;
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::drain_loop() {
  try {
    while (system_state_ != kShuttingDownScheduleLoop) {
      drain_loop_prelude();
      LoadQueueItem item;
      if (load_queue_item(item)) {
        drain_batch(item);
      }
    }
  } catch (const ShutDownException&) {
    // This is expected when the state loop is shutting down.
    LOG_DEBUG("{}::{}() caught ShutDownException, exiting...", kCls, __func__);
  }

  system_state_ = kShutDownComplete;
  in_drain_loop_prelude_ = false;
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::state_loop() {
  const char* func = __func__;
  mit::unique_lock lock(main_mutex_);
  while (true) {
    cv_main_.wait(lock, [&] {
      if (system_state_ == kPaused) {
        LOG_INFO("{}::{}() done waiting @{}", kCls, func, __LINE__);
        return true;
      }
      if (num_connections_ == 0) {
        if (mcts::kEnableServiceDebug) {
          LOG_INFO("{}::{}() exiting @{}", kCls, func, __LINE__);
        }
        return true;
      }
      if (mcts::kEnableServiceDebug) {
        LOG_INFO("{}::{}() waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
      }
      return false;
    });

    if (num_connections_ == 0) {
      break;
    }

    LOG_INFO("{}::{}() state={} @{}", kCls, func, system_state_, __LINE__);
    RELEASE_ASSERT(system_state_ == kPaused, "Unexpected system_state: {} (expected {})",
                   system_state_, kPaused);

    core::LoopControllerClient::get()->handle_pause_receipt(__FILE__, __LINE__);

    cv_main_.wait(lock, [&] {
      if (system_state_ == kUnpaused) {
        LOG_INFO("{}::{}() done waiting @{}", kCls, func, __LINE__);
        return true;
      }
      if (num_connections_ == 0) {
        if (mcts::kEnableServiceDebug) {
          LOG_INFO("{}::{}() exiting @{}", kCls, func, __LINE__);
        }
        return true;
      }
      LOG_INFO("{}::{}() waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
      return false;
    });

    if (num_connections_ == 0) {
      break;
    }

    LOG_INFO("{}::{}() state={} @{}", kCls, func, system_state_, __LINE__);
    RELEASE_ASSERT(system_state_ == kUnpaused, "Unexpected system_state: {} (expected {})",
                   system_state_, kUnpaused);

    // wait for schedule-loop/drain-loop to exit their preludes
    cv_main_.wait(lock, [&] {
      if (in_schedule_loop_prelude_ || in_drain_loop_prelude_) {
        LOG_INFO("{}::{}() still waiting... (in_schedule={}, in_drain={}) @{}", kCls, func,
                 in_schedule_loop_prelude_, in_drain_loop_prelude_, __LINE__);
        return false;
      }
      return true;
    });

    core::LoopControllerClient::get()->handle_unpause_receipt(__FILE__, __LINE__);
  }

  system_state_ = kShuttingDownScheduleLoop;
  lock.unlock();
  cv_main_.notify_all();
  lock.lock();

  if (mcts::kEnableServiceDebug) {
    LOG_INFO("{}::{}() state={} @{}", kCls, func, system_state_, __LINE__);
  }
  cv_main_.wait(lock, [&] {
    if (system_state_ == kShutDownComplete) {
      if (mcts::kEnableServiceDebug) {
        LOG_INFO("{}::{}() done waiting @{}", kCls, func, __LINE__);
      }
      return true;
    }
    if (mcts::kEnableServiceDebug) {
      LOG_INFO("{}::{}() waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
    }
    return false;
  });
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("{}::{}() done!", kCls, func);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::load_initial_weights_if_necessary() {
  if (ready_) return;

  auto client = core::LoopControllerClient::get();
  if (!client) {
    if (initial_weights_loaded_) {
      ready_ = true;
      return;
    } else {
      throw util::CleanException(
        "If --loop-controller-port= is not specified, then must specify either --model-filename/-m "
        "or --no-model");
    }
  }

  LOG_INFO("{}: handling worker-ready...", kCls);
  client->handle_worker_ready();
  mit::unique_lock lock(main_mutex_);
  cv_main_.wait(lock, [&] {
    return initial_weights_loaded_ || system_state_ == kShuttingDownScheduleLoop ||
           system_state_ == kPausingScheduleLoop;
  });

  if (system_state_ == kShuttingDownScheduleLoop) throw ShutDownException();

  if (initial_weights_loaded_) {
    ready_ = true;
    LOG_INFO("{}: weights loaded!", kCls);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::schedule_loop_prelude() {
  if (system_state_ == kUnpaused) return;  // early exit for common case, bypassing lock

  const char* func = __func__;
  mit::unique_lock lock(main_mutex_);

  in_schedule_loop_prelude_ = true;
  LOG_INFO("{}::{}() ({}) @{}", kCls, func, system_state_, __LINE__);

  if (system_state_ == kShuttingDownScheduleLoop) throw ShutDownException();
  RELEASE_ASSERT(system_state_ == kPausingScheduleLoop, "Unexpected system_state: {} (expected {})",
                 system_state_, kPausingScheduleLoop);
  system_state_ = kPausingDrainLoop;
  lock.unlock();
  cv_main_.notify_all();

  lock.lock();
  cv_main_.wait(lock, [&] {
    if (system_state_ == kShuttingDownScheduleLoop || system_state_ == kUnpausingScheduleLoop) {
      return true;
    }
    LOG_INFO("{}::{}() still waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
    return false;
  });

  if (system_state_ == kShuttingDownScheduleLoop) throw ShutDownException();
  system_state_ = kUnpausingDrainLoop;
  LOG_INFO("{}::{}() ({}) @{}", kCls, func, system_state_, __LINE__);

  lock.unlock();
  cv_main_.notify_all();

  lock.lock();
  cv_main_.wait(lock, [&] {
    if (system_state_ == kShuttingDownScheduleLoop || system_state_ == kUnpaused) {
      return true;
    }
    LOG_INFO("{}::{}() still waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
    return false;
  });

  if (system_state_ == kShuttingDownScheduleLoop) throw ShutDownException();
  LOG_INFO("{}::{}() ({}) @{}", kCls, func, system_state_, __LINE__);

  in_schedule_loop_prelude_ = false;
  lock.unlock();
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::drain_loop_prelude() {
  if (system_state_ == kUnpaused) return;  // early exit for common case, bypassing lock

  const char* func = __func__;
  mit::unique_lock lock(main_mutex_);
  in_drain_loop_prelude_ = true;
  cv_main_.wait(lock, [&] {
    if (system_state_ == kShuttingDownDrainLoop || system_state_ == kPausingDrainLoop) {
      return true;
    }
    LOG_INFO("{}::{}() still waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
    return false;
  });

  LOG_INFO("{}::{}() (state={} queue={}) @{}", kCls, func, system_state_, load_queue_.size(),
           __LINE__);

  if (system_state_ == kShuttingDownDrainLoop) throw ShutDownException();

  if (load_queue_.empty()) {
    system_state_ = kPaused;
    LOG_INFO("{}::{}() ({}) @{}", kCls, func, system_state_, __LINE__);
    lock.unlock();
    cv_main_.notify_all();

    lock.lock();
    cv_main_.wait(lock, [&] {
      if (system_state_ == kShuttingDownDrainLoop || system_state_ == kUnpausingDrainLoop) {
        return true;
      }
      LOG_INFO("{}::{}() still waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
      return false;
    });

    if (system_state_ == kShuttingDownDrainLoop) throw ShutDownException();
    system_state_ = kUnpaused;
  }

  in_drain_loop_prelude_ = false;
  LOG_INFO("{}::{}() ({}) @{}", kCls, func, system_state_, __LINE__);
  lock.unlock();
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::BatchData* NNEvaluationService<Game>::get_next_batch_data(
  core::NNEvalScheduleLoopPerfStats& schedule_loop_stats) {
  mit::unique_lock lock(main_mutex_);
  core::PerfClocker clocker(schedule_loop_stats.wait_for_search_threads_time_ns);

  const char* func = __func__;
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}()", kCls, func);
  }

  auto predicate = [&] {
    if (system_state_ == kPausingScheduleLoop || system_state_ == kShuttingDownScheduleLoop) {
      return true;
    }

    BatchData* batch_data = batch_data_slice_allocator_.get_first_pending_batch_data();
    if (batch_data) {
      if (batch_data->frozen()) {
        if (mcts::kEnableServiceDebug) {
          LOG_INFO("<-- {}::{}() (count:{})", kCls, func, batch_data->allocate_count);
        }
        return true;
      }
      if (mcts::kEnableServiceDebug) {
        LOG_INFO("<-- {}::{}() still waiting (seq:{} accepting:{} alloc:{} write:{})", kCls, func,
                 batch_data->sequence_id, batch_data->accepting_allocations,
                 batch_data->allocate_count, batch_data->write_count);
      }
    }
    if (mcts::kEnableServiceDebug) {
      LOG_INFO("<-- {}::{}() still waiting (no batch data)", kCls, func);
    }
    return false;
  };

  cv_main_.wait(lock, predicate);
  if (system_state_ == kPausingScheduleLoop || system_state_ == kShuttingDownScheduleLoop) {
    return nullptr;
  }
  return batch_data_slice_allocator_.pop_first_pending_batch_data();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::schedule_batch(
  BatchData* batch_data, core::NNEvalScheduleLoopPerfStats& schedule_loop_stats) {
  if (!batch_data) return;
  RELEASE_ASSERT(batch_data->frozen());

  const char* func = __func__;
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}() (service:{} seq:{}, count:{})", kCls, func, this->instance_id_,
             batch_data->sequence_id, batch_data->allocate_count);
  }

  core::PerfClocker pipeline_wait_clocker(schedule_loop_stats.pipeline_wait_time_ns);
  core::pipeline_index_t pipeline_index = net_.get_pipeline_assignment();

  core::PerfClocker pipeline_schedule_clocker(pipeline_wait_clocker,
                                              schedule_loop_stats.pipeline_schedule_time_ns);
  int num_rows = batch_data->write_count;
  batch_data->copy_input_to(num_rows, net_, pipeline_index);
  net_.schedule(pipeline_index);
  pipeline_schedule_clocker.stop();

  mit::unique_lock lock(main_mutex_);
  load_queue_.emplace(batch_data, pipeline_index);
  lock.unlock();
  cv_main_.notify_all();

  int max_size = batch_data_slice_allocator_.batch_size_limit();
  schedule_loop_stats.positions_evaluated = num_rows;
  schedule_loop_stats.batches_evaluated = 1;
  schedule_loop_stats.full_batches_evaluated = num_rows == max_size ? 1 : 0;

  mit::unique_lock perf_lock(perf_stats_mutex_);
  perf_stats_.update(schedule_loop_stats);
}

template <core::concepts::Game Game>
bool NNEvaluationService<Game>::load_queue_item(LoadQueueItem& item) {
  const char* func = __func__;
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}() - acquiring load_queue_mutex_ (service:{})", kCls, func,
             this->instance_id_);
  }
  mit::unique_lock lock(main_mutex_);
  cv_main_.wait(lock, [&] {
    if (!load_queue_.empty() || system_state_ == kPausingDrainLoop ||
        system_state_ == kShuttingDownDrainLoop) {
      if (mcts::kEnableServiceDebug) {
        LOG_INFO("<-- {}::{}() - done waiting! (service:{}, state:{}, queue:{})", kCls, func,
                 this->instance_id_, system_state_, load_queue_.size());
      }
      return true;
    }

    if (mcts::kEnableServiceDebug) {
      LOG_INFO("<-- {}::{}() - still waiting... (service:{}, state:{}, queue:{})", kCls, func,
               this->instance_id_, system_state_, load_queue_.size());
    }
    return false;
  });

  if (system_state_ == kShuttingDownDrainLoop) throw ShutDownException();
  if (load_queue_.empty()) return false;

  item = load_queue_.front();
  load_queue_.pop();
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}() - returning item (service:{} seq:{}, pipeline_index:{})", kCls, func,
             this->instance_id_, item.batch_data->sequence_id, item.pipeline_index);
  }
  return true;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::drain_batch(const LoadQueueItem& item) {
  const char* func = __func__;
  BatchData* batch_data = item.batch_data;
  core::pipeline_index_t pipeline_index = item.pipeline_index;

  float* policy_data;
  float* value_data;
  float* action_values_data;

  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}() - loading (service:{} seq:{} pipeline_index:{})", kCls, func,
             this->instance_id_, batch_data->sequence_id, pipeline_index);
  }

  net_.load(pipeline_index, &policy_data, &value_data, &action_values_data);
  batch_data->load(policy_data, value_data, action_values_data);
  net_.release(pipeline_index);

  mit::unique_lock lock(main_mutex_);
  yield_manager_->notify(batch_data->notification_tasks);
  RELEASE_ASSERT(last_evaluated_sequence_id_ < batch_data->sequence_id);
  last_evaluated_sequence_id_ = batch_data->sequence_id;
  batch_data_slice_allocator_.recycle(batch_data);

  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}() - (service:{} seq:{}) complete!", kCls, func, this->instance_id_,
             last_evaluated_sequence_id_);
  }

  lock.unlock();
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::reload_weights(const std::vector<char>& buf) {
  const char* func = __func__;
  LOG_INFO("{}: reloading network weights...", kCls);
  RELEASE_ASSERT(system_state_ == kPaused, "{}() called while not paused", func);

  std::ispanstream stream{std::span<const char>(buf)};
  mit::unique_lock lock(main_mutex_);
  net_.deactivate();
  net_.load_weights(stream);
  initial_weights_loaded_ = true;
  lock.unlock();
  cv_main_.notify_all();

  LOG_INFO("{}: clearing network cache...", kCls);

  // TODO: we can clear each shard's cache in parallel for a slight performance boost
  for (int i = 0; i < kNumHashShards; ++i) {
    ShardData& shard_data = shard_datas_[i];
    mit::unique_lock shard_lock(shard_data.mutex);
    shard_data.eval_cache.clear();
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::pause() {
  const char* func = __func__;
  LOG_INFO("{}::{}()", kCls, func);
  mit::unique_lock lock(main_mutex_);
  system_state_ = kPausingScheduleLoop;
  lock.unlock();
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::unpause() {
  const char* func = __func__;
  LOG_INFO("{}::{}() [state:{}]", kCls, func, system_state_);
  mit::unique_lock lock(main_mutex_);

  // we wait in case we are already in the middle of a pause/unpause operation
  cv_main_.wait(lock, [&] {
    if (system_state_ == kPaused || system_state_ == kUnpaused ||
        system_state_ == kShutDownComplete) {
      LOG_INFO("{}::{}() done waiting ({}) @{}", kCls, func, system_state_, __LINE__);
      return true;
    }
    LOG_INFO("{}::{}() still waiting... ({}) @{}", kCls, func, system_state_, __LINE__);
    return false;
  });
  system_state_ = kUnpausingScheduleLoop;
  lock.unlock();
  cv_main_.notify_all();
}

}  // namespace mcts
