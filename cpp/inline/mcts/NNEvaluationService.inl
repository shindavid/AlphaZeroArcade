#include <mcts/NNEvaluationService.hpp>

#include <util/Asserts.hpp>
#include <util/KeyValueDumper.hpp>
#include <util/LoggingUtil.hpp>

#include <boost/json/src.hpp>

#include <cstdint>
#include <spanstream>

namespace mcts {

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::instance_map_t
    NNEvaluationService<Game>::instance_map_;

template <core::concepts::Game Game>
int NNEvaluationService<Game>::instance_count_ = 0;

template <core::concepts::Game Game>
NNEvaluationService<Game>* NNEvaluationService<Game>::create(
    const NNEvaluationServiceParams& params, core::GameServerBase* server) {
  auto it = instance_map_.find(params.model_filename);
  if (it == instance_map_.end()) {
    auto instance = new NNEvaluationService(params, server);

    instance_map_[params.model_filename] = instance;
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
  std::lock_guard<std::mutex> guard(connection_mutex_);
  num_connections_++;
  if (thread_) return;

  thread_ = new std::thread([&] { this->loop(); });
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::disconnect() {
  std::lock_guard<std::mutex> guard(connection_mutex_);
  if (thread_) {
    num_connections_--;
    if (num_connections_ > 0) return;
    cv_net_weights_.notify_all();
    cv_main_.notify_all();
    if (thread_->joinable()) thread_->join();
    delete thread_;
    thread_ = nullptr;
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::CacheLookupResult::update_notifying_batch_data(
  BatchData* batch_data) {
  if (!notifying_batch_data || batch_data->sequence_id > notifying_batch_data->sequence_id) {
    notifying_batch_data = batch_data;
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
      core::GameServerClient(),
      instance_id_(instance_count_++),
      params_(params),
      num_game_threads_(server->num_game_threads()),
      full_input_(util::to_std_array<int64_t>(params.batch_size_limit,
                                              eigen_util::to_int64_std_array_v<InputShape>)),
      batch_data_slice_allocator_(params.batch_size_limit, perf_stats_) {
  if (!params.model_filename.empty()) {
    net_.load_weights(params.model_filename.c_str(), params.cuda_device);
    net_.activate();
  }
  auto input_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                 eigen_util::to_int64_std_array_v<InputShape>);
  auto policy_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                  eigen_util::to_int64_std_array_v<PolicyShape>);
  auto value_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                 eigen_util::to_int64_std_array_v<ValueShape>);
  auto action_value_shape = util::to_std_array<int64_t>(
      params_.batch_size_limit, eigen_util::to_int64_std_array_v<ActionValueShape>);

  torch_input_gpu_ = torch::empty(input_shape, torch_util::to_dtype_v<float>)
                         .to(at::Device(params.cuda_device));
  torch_policy_ = torch::empty(policy_shape, torch_util::to_dtype_v<float>);
  torch_value_ = torch::empty(value_shape, torch_util::to_dtype_v<float>);
  torch_action_value_ = torch::empty(action_value_shape, torch_util::to_dtype_v<float>);

  input_vec_.push_back(torch_input_gpu_);

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
void NNEvaluationService<Game>::TensorGroup::load_output_from(int row, torch::Tensor& torch_policy,
                                                              torch::Tensor& torch_value,
                                                              torch::Tensor& torch_action_value) {
  constexpr size_t policy_size = PolicyShape::total_size;
  constexpr size_t value_size = ValueShape::total_size;
  constexpr size_t action_value_size = ActionValueShape::total_size;

  memcpy(policy.data(), torch_policy.data_ptr<float>() + row * policy_size,
         policy_size * sizeof(float));
  memcpy(value.data(), torch_value.data_ptr<float>() + row * value_size,
         value_size * sizeof(float));
  memcpy(action_values.data(), torch_action_value.data_ptr<float>() + row * action_value_size,
         action_value_size * sizeof(float));
}

template <core::concepts::Game Game>
NNEvaluationService<Game>::BatchData::BatchData(int capacity) : tensor_groups(capacity) {}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchData::copy_input_to(int num_rows,
                                                         DynamicInputTensor& full_input) {
  float* full_input_data = full_input.data();
  constexpr size_t input_size = InputShape::total_size;
  int r = 0;
  for (int row = 0; row < num_rows; row++) {
    const TensorGroup& group = tensor_groups[row];
    memcpy(full_input_data + r, group.input.data(), input_size * sizeof(float));
    r += input_size;
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
  int batch_size_limit, core::PerfStats& perf_stats)
    : batch_size_limit_(batch_size_limit), perf_stats_(perf_stats) {
  add_batch_data();
}

template <core::concepts::Game Game>
NNEvaluationService<Game>::BatchDataSliceAllocator::~BatchDataSliceAllocator() {
  while (!pending_batch_datas_.empty()) {
    BatchData* batch_data = pending_batch_datas_.front();
    pending_batch_datas_.pop();
    delete batch_data;
  }

  for (BatchData* batch_data : batch_data_reserve_) {
    delete batch_data;
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchDataSliceAllocator::allocate_slices(BatchDataSlice* slices,
                                                                         int n,
                                                                         std::mutex& main_mutex) {
  std::unique_lock lock(main_mutex);

  int slice_index = 0;
  BatchData* batch_data = pending_batch_datas_.back();
  while (n) {
    BatchDataSlice& slice = slices[slice_index++];

    int m = std::min(n, batch_data->capacity() - batch_data->allocate_count);
    util::release_assert(m > 0);
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
bool NNEvaluationService<Game>::BatchDataSliceAllocator::freeze_first() {
  if (pending_batch_datas_.empty()) return false;
  BatchData* batch_data = pending_batch_datas_.front();
  if (batch_data->allocate_count == 0) return false;

  LOG_DEBUG("<-- NNEvaluationService: Freezing batch data {} (alloc:{} write:{})",
            batch_data->sequence_id, batch_data->allocate_count, batch_data->write_count);

  bool was_frozen = batch_data->frozen();
  batch_data->accepting_allocations = false;
  bool newly_frozen = batch_data->frozen() && !was_frozen;

  if (pending_batch_datas_.size() == 1) {
    add_batch_data();
  }

  return newly_frozen;
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
    pending_batch_datas_.pop();
    if (pending_batch_datas_.empty()) {
      add_batch_data();
    }
    return batch_data;
  }
  return nullptr;
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::BatchData*
NNEvaluationService<Game>::BatchDataSliceAllocator::add_batch_data() {
  // Assumes mutex_ is locked
  BatchData* batch_data;
  if (batch_data_reserve_.empty()) {
    batch_data = new BatchData(batch_size_limit_);
    perf_stats_.nn_eval_loop_stats.batch_datas_allocated++;
  } else {
    batch_data = batch_data_reserve_.back();
    batch_data_reserve_.pop_back();
  }
  batch_data->sequence_id = next_batch_data_sequence_id_++;
  pending_batch_datas_.push(batch_data);
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
      result.update_notifying_batch_data(last_miss_info.batch_data);
    }

    // notifying_batch_data is either set above because cache_misses > 0, or it is set inside
    // handle_fresh_item() in the hit_cache && eval->pending() case. There is no other way that
    // can_continue can be set to false.
    util::debug_assert(result.notifying_batch_data != nullptr);
    register_notification_task(request, result.notifying_batch_data);

    for (int i = 0; i < result.stats.cache_misses; ++i) {
      CacheMissInfo& miss_info = miss_infos[i];
      RequestItem& item = request.get_fresh_item(miss_info.item_index);
      BatchData* batch_data = miss_info.batch_data;
      int row = miss_info.row;
      write_to_batch(item, batch_data, row);
    }

    yield_instruction = core::kYield;
  }

  std::unique_lock perf_stats_lock(perf_stats_mutex_);
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

  int64_t positions_evaluated = stats.nn_eval_loop_stats.positions_evaluated;
  int64_t batches_evaluated = stats.nn_eval_loop_stats.batches_evaluated;
  int64_t full_batches_evaluated = stats.nn_eval_loop_stats.full_batches_evaluated;

  int64_t wait_for_game_slot_time_ns = stats.search_thread_stats.wait_for_game_slot_time_ns;
  int64_t cache_mutex_acquire_time_ns = stats.search_thread_stats.cache_mutex_acquire_time_ns;
  int64_t cache_insert_time_ns = stats.search_thread_stats.cache_insert_time_ns;
  int64_t batch_prepare_time_ns = stats.search_thread_stats.batch_prepare_time_ns;
  int64_t batch_write_time_ns = stats.search_thread_stats.batch_write_time_ns;
  int64_t wait_for_nn_eval_time_ns = stats.search_thread_stats.wait_for_nn_eval_time_ns;
  int64_t mcts_time_ns = stats.search_thread_stats.mcts_time_ns;

  int64_t wait_for_search_threads_time_ns =
    stats.nn_eval_loop_stats.wait_for_search_threads_time_ns;
  int64_t cpu2gpu_copy_time_ns = stats.nn_eval_loop_stats.cpu2gpu_copy_time_ns;
  int64_t gpu2cpu_copy_time_ns = stats.nn_eval_loop_stats.gpu2cpu_copy_time_ns;
  int64_t model_eval_time_ns = stats.nn_eval_loop_stats.model_eval_time_ns;

  int batch_datas_allocated = stats.nn_eval_loop_stats.batch_datas_allocated;

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
  float per_batch_cpu2gpu_copy_time_ms = 1e-6 * cpu2gpu_copy_time_ns / batches_evaluated;
  float per_batch_gpu2cpu_copy_time_ms = 1e-6 * gpu2cpu_copy_time_ns / batches_evaluated;
  float per_batch_model_eval_time_ms = 1e-6 * model_eval_time_ns / batches_evaluated;
  float per_pos_model_eval_time_us = 1e-3 * model_eval_time_ns / positions_evaluated;

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
  dump("nn-eval per-batch cpu2gpu copy time", "%.3fms", per_batch_cpu2gpu_copy_time_ms);
  dump("nn-eval per-batch gpu2cpu copy time", "%.3fms", per_batch_gpu2cpu_copy_time_ms);
  dump("nn-eval per-batch model eval time", "%.3fms", per_batch_model_eval_time_ms);
  dump("nn-eval per-pos model eval time", "%.3fus", per_pos_model_eval_time_us);
  session_ended_ = true;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::update_perf_stats(core::PerfStats& perf_stats) {
  std::unique_lock lock(perf_stats_mutex_);
  core::PerfStats perf_stats_copy = perf_stats_;
  new (&perf_stats_) core::PerfStats();
  lock.unlock();

  perf_stats += perf_stats_copy;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::handle_force_progress() {
  std::unique_lock lock(main_mutex_);
  LOG_DEBUG("<-- NNEvaluationService::{}() size={}", __func__,
            batch_data_slice_allocator_.pending_batch_datas_size());

  if (batch_data_slice_allocator_.freeze_first()) {
    lock.unlock();
    cv_main_.notify_all();
  }
}

template <core::concepts::Game Game>
std::string NNEvaluationService<Game>::dump_key(const char* descr) {
  return std::format("NN-{} {}", instance_id_, descr);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::check_cache(NNEvaluationRequest& request,
                                            CacheLookupResult& result) {
  int m = request.num_stale_items();
  int n = request.num_fresh_items();
  int s = m + n;
  util::debug_assert(n > 0);

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
  util::debug_assert(n > 0);

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

  util::debug_assert(s == m + n);

  // Now use std::sort() to group by hash shard.
  std::sort(sort_items, sort_items + s);
}

template <core::concepts::Game Game>
bool NNEvaluationService<Game>::handle_fresh_item(NNEvaluationRequest& request,
                                                  CacheLookupResult& result, ShardData& shard,
                                                  int item_index) {
  core::PerfClocker clocker(result.stats.cache_insert_time_ns);
  RequestItem& item = request.get_fresh_item(item_index);
  util::release_assert(item.eval() == nullptr);

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
      result.update_notifying_batch_data(eval_data);
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

  int B = params_.batch_size_limit;
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

    item.eval()->set_aux(slice.batch_data);
    result.update_notifying_batch_data(slice.batch_data);
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

  std::unique_lock lock(main_mutex_);
  batch_data->write_count++;
  bool notify = batch_data->frozen();
  lock.unlock();
  if (notify) {
    cv_main_.notify_all();
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::register_notification_task(const NNEvaluationRequest& request,
                                                           BatchData* batch_data) {
  // NOTE: in principle, we can initialize yield_manager_ at startup to avoid doing it here.
  // There should only ever be one yield_manager_ for the entire process. We do it here to
  // avoid having to pass it around all over the place during initialization.
  const core::YieldNotificationUnit& unit = request.notification_unit();
  if (this->yield_manager_ == nullptr) {
    yield_manager_ = unit.yield_manager;
  } else {
    // Verify that there is only one
    util::debug_assert(yield_manager_ == unit.yield_manager);
  }

  std::unique_lock lock(main_mutex_);
  batch_data->notification_tasks.push_back(unit.slot_context());
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::loop() {
  while (active()) {
    core::NNEvalLoopPerfStats loop_stats;

    load_initial_weights_if_necessary();
    wait_for_unpause();
    BatchData* batch_data = get_next_batch_data(loop_stats);
    batch_evaluate(batch_data, loop_stats);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::load_initial_weights_if_necessary() {
  if (ready_) return;
  ready_ = true;

  auto client = core::LoopControllerClient::get();
  if (!client) {
    if (initial_weights_loaded_) {
      return;
    } else {
      throw util::CleanException(
        "If --loop-controller-port= is not specified, then must specify either --model-filename/-m "
        "or --no-model");
    }
  }

  LOG_INFO("NNEvaluationService: sending worker-ready...");

  client->send_worker_ready();
  std::unique_lock<std::mutex> net_weights_lock(net_weights_mutex_);
  cv_net_weights_.wait(net_weights_lock, [&] { return initial_weights_loaded_ || !active(); });
  LOG_INFO("NNEvaluationService: weights loaded!");
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_for_unpause() {
  if (!skip_next_pause_receipt_ && !paused_) return;  // early exit for common case, bypassing lock

  LOG_INFO("NNEvaluationService: wait_for_unpause - acquiring main_mutex_");
  std::unique_lock lock(main_mutex_);
  if (skip_next_pause_receipt_) {
    LOG_INFO("NNEvaluationService: skipping handle_pause_receipt");
    skip_next_pause_receipt_ = false;
  } else {
    net_.deactivate();
    LOG_INFO("NNEvaluationService: handle_pause_receipt");
    core::LoopControllerClient::get()->handle_pause_receipt();
  }
  cv_main_.wait(lock, [&] { return !paused_; });
  lock.unlock();

  LOG_INFO("NNEvaluationService: handle_unpause_receipt");
  core::LoopControllerClient::get()->handle_unpause_receipt();
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::BatchData* NNEvaluationService<Game>::get_next_batch_data(
  core::NNEvalLoopPerfStats& loop_stats) {
  std::unique_lock lock(main_mutex_);
  core::PerfClocker clocker(loop_stats.wait_for_search_threads_time_ns);

  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}()", cls, func);
  }

  auto predicate = [&] {
    if (!active() || paused_) return true;
    BatchData* batch_data = batch_data_slice_allocator_.get_first_pending_batch_data();
    if (batch_data) {
      if (batch_data->frozen()) {
        if (mcts::kEnableServiceDebug) {
          LOG_INFO("<-- {}::{}() (count:{})", cls, func, batch_data->allocate_count);
        }
        return true;
      }
      if (mcts::kEnableServiceDebug) {
        LOG_INFO("<-- {}::{}() still waiting (seq:{} accepting:{} alloc:{} write:{})", cls, func,
                 batch_data->sequence_id, batch_data->accepting_allocations,
                 batch_data->allocate_count, batch_data->write_count);
      }
    }
    if (mcts::kEnableServiceDebug) {
      LOG_INFO("<-- {}::{}() still waiting (no batch data)", cls, func);
    }
    return false;
  };

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  cv_main_.wait_until(lock, deadline, predicate);
  bool deadline_reached = std::chrono::steady_clock::now() >= deadline;
  if (deadline_reached) {
    // If this happens, there is some sort of bug. Retrying here potentially covers up for the bug.
    LOG_WARN("<-- {}::{}() Timed out waiting for batch data. Indicates a bug!", cls, func);
    BatchData* batch_data = batch_data_slice_allocator_.get_first_pending_batch_data();
    if (!batch_data) {
      return nullptr;
    } else {
      LOG_WARN("<-- {}::{}() Retrying...", cls, func);
      batch_data->accepting_allocations = false;
      cv_main_.wait(lock, predicate);
    }
  }
  if (!active() || paused_) return nullptr;
  return batch_data_slice_allocator_.pop_first_pending_batch_data();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::batch_evaluate(BatchData* batch_data,
                                               core::NNEvalLoopPerfStats& loop_stats) {
  if (!batch_data) return;
  util::release_assert(batch_data->frozen());

  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}() (seq:{}, count:{})", cls, func, batch_data->sequence_id,
             batch_data->allocate_count);
  }

  // NOTE: we could slightly optimize this by moving some of these steps out of the main path.
  // If the batch is already filled up before we get to this point, we can start copying to
  // full_input_ in a separate thread, so that it's ready when we get here.
  //
  // Furthermore, that thread could also start copying the input to the GPU. We would need to make
  // sure that we have two separate tensors on the GPU, so that we can copy to one while
  // evaluating the other. This would require some extra synchronization, and would also limit the
  // potential batch size by half, and would also mean that the GPU data memory address would change
  // every time we switch the input tensor. So it's unclear if this is worth it.
  //
  // Something similar could potentially be done when copying output back to the CPU.
  //
  // Currently, on dshin's laptop, for a 10,000-game benchmark run of c4, we see:
  //
  // NN-0 nn-eval per-batch cpu2gpu time:                 0.061ms
  // NN-0 nn-eval per-batch gpu2cpu time:                 0.014ms
  // NN-0 nn-eval per-batch model eval time:              3.243ms
  //
  // These numbers put a limit on the potential speedup we could get from this optimization.
  core::PerfClocker gpu_copy_clocker(loop_stats.cpu2gpu_copy_time_ns);
  int num_rows = batch_data->write_count;
  batch_data->copy_input_to(num_rows, full_input_);
  auto input_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                 eigen_util::to_int64_std_array_v<InputShape>);
  torch::Tensor full_input_torch = torch::from_blob(full_input_.data(), input_shape);
  torch_input_gpu_.copy_(full_input_torch);

  core::PerfClocker model_eval_clocker(gpu_copy_clocker, loop_stats.model_eval_time_ns);
  net_.predict(input_vec_, torch_policy_, torch_value_, torch_action_value_);

  core::PerfClocker gpu_copy_clocker2(model_eval_clocker, loop_stats.gpu2cpu_copy_time_ns);
  for (int i = 0; i < num_rows; ++i) {
    TensorGroup& group = batch_data->tensor_groups[i];
    group.load_output_from(i, torch_policy_, torch_value_, torch_action_value_);
    group.eval->init(group.value, group.policy, group.action_values, group.valid_actions, group.sym,
                     group.active_seat, group.action_mode);
  }
  yield_manager_->notify(batch_data->notification_tasks);
  gpu_copy_clocker2.stop();

  std::unique_lock lock(main_mutex_);
  util::release_assert(last_evaluated_sequence_id_ < batch_data->sequence_id);
  last_evaluated_sequence_id_ = batch_data->sequence_id;
  batch_data_slice_allocator_.recycle(batch_data);
  lock.unlock();
  cv_main_.notify_all();

  loop_stats.positions_evaluated = num_rows;
  loop_stats.batches_evaluated = 1;
  loop_stats.full_batches_evaluated = num_rows == params_.batch_size_limit ? 1 : 0;

  std::unique_lock perf_lock(perf_stats_mutex_);
  perf_stats_.update(loop_stats);

  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<-- {}::{}() - (seq:{}) complete!", cls, func, last_evaluated_sequence_id_);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::reload_weights(const std::vector<char>& buf,
                                               const std::string& cuda_device) {
  LOG_INFO("NNEvaluationService: reloading network weights...");
  util::release_assert(paused_, "{}() called while not paused", __func__);

  std::ispanstream stream{std::span<const char>(buf)};
  std::unique_lock net_weights_lock(net_weights_mutex_);
  net_.load_weights(stream, cuda_device);
  initial_weights_loaded_ = true;
  net_weights_lock.unlock();
  cv_net_weights_.notify_all();

  LOG_INFO("NNEvaluationService: clearing network cache...");

  // TODO: we can clear each shard's cache in parallel for a slight performance boost
  for (int i = 0; i < kNumHashShards; ++i) {
    ShardData& shard_data = shard_datas_[i];
    std::unique_lock shard_lock(shard_data.mutex);
    shard_data.eval_cache.clear();
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::pause() {
  LOG_INFO("NNEvaluationService: pausing");
  std::unique_lock lock(main_mutex_);
  if (paused_) {
    net_.deactivate();
    LOG_INFO("NNEvaluationService: handle_pause_receipt (already paused)");
    core::LoopControllerClient::get()->handle_pause_receipt();
    return;
  }
  paused_ = true;

  if (!initial_weights_loaded_) {
    net_.deactivate();
    skip_next_pause_receipt_ = true;
    LOG_INFO("NNEvaluationService: handle_pause_receipt (skip next)");
    core::LoopControllerClient::get()->handle_pause_receipt();
  }
  LOG_INFO("NNEvaluationService: pause complete!");

  lock.unlock();
  cv_main_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::unpause() {
  LOG_INFO("NNEvaluationService: unpausing");
  std::unique_lock lock(main_mutex_);
  net_.activate();
  if (!paused_) {
    LOG_INFO("NNEvaluationService: handle_unpause_receipt (already unpaused)");
    core::LoopControllerClient::get()->handle_unpause_receipt();
    return;
  }
  paused_ = false;
  lock.unlock();
  cv_main_.notify_all();
  LOG_INFO("NNEvaluationService: unpause complete!");
}

}  // namespace mcts
