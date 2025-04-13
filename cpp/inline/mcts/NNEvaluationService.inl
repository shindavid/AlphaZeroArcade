#include <mcts/NNEvaluationService.hpp>

#include <core/BasicTypes.hpp>
#include <core/PerfStats.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluationServiceBase.hpp>
#include <util/Asserts.hpp>
#include <util/KeyValueDumper.hpp>

#include <boost/json/src.hpp>

#include <mutex>
#include <spanstream>

namespace mcts {

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::instance_map_t
    NNEvaluationService<Game>::instance_map_;

template <core::concepts::Game Game>
int NNEvaluationService<Game>::instance_count_ = 0;

template <core::concepts::Game Game>
NNEvaluationService<Game>* NNEvaluationService<Game>::create(
    const NNEvaluationServiceParams& params) {
  auto it = instance_map_.find(params.model_filename);
  if (it == instance_map_.end()) {
    auto instance = new NNEvaluationService(params);

    if (mcts::kEnableProfiling) {
      instance->set_profiling_dir(params.profiling_dir());
    }
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
    cv_eval_.notify_all();
    if (thread_->joinable()) thread_->join();
    delete thread_;
    thread_ = nullptr;
  }
  profiler_.close_file();
}

template <core::concepts::Game Game>
inline void NNEvaluationService<Game>::set_profiling_dir(
    const boost::filesystem::path& profiling_dir) {
  std::string name = util::create_string("eval-%d", instance_id_);
  auto profiling_file_path = profiling_dir / util::create_string("%s.txt", name.c_str());
  profiler_.initialize_file(profiling_file_path);
  profiler_.set_name(name);
}

template <core::concepts::Game Game>
inline NNEvaluationService<Game>::NNEvaluationService(const NNEvaluationServiceParams& params)
    : instance_id_(instance_count_++),
      params_(params),
      full_input_(util::to_std_array<int64_t>(params.batch_size_limit,
                                              eigen_util::to_int64_std_array_v<InputShape>)),
      timeout_duration_(params.nn_eval_timeout_ns),
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
  deadline_ = std::chrono::steady_clock::now();

  int n = mcts::kNumCacheShards;
  for (int i = 0; i < n; ++i) {
    int c = params_.cache_size;
    size_t capacity = c * (i + 1) / n - c * i / n;

    Shard& shard = shards_[i];
    shard.eval_cache.set_capacity(capacity);
    shard.eval_cache.set_eviction_handler([&](NNEvaluation* e) { decrement_ref_count(shard, e); });
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
void NNEvaluationService<Game>::TensorGroup::load_output_from(
    int row, torch::Tensor& torch_policy, torch::Tensor& torch_value,
    torch::Tensor& torch_action_value) {
  constexpr size_t policy_size = PolicyShape::total_size;
  constexpr size_t value_size = ValueShape::total_size;
  constexpr size_t action_value_size = ActionValueShape::total_size;

  memcpy(policy.data(), torch_policy.data_ptr<float>() + row * policy_size,
         policy_size * sizeof(float));
  memcpy(value.data(), torch_value.data_ptr<float>() + row * value_size,
         value_size * sizeof(float));
  memcpy(action_values.data(),
         torch_action_value.data_ptr<float>() + row * action_value_size,
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
  frozen_allocate_count = -1;
}

template <core::concepts::Game Game>
bool NNEvaluationService<Game>::BatchData::freeze() {
  bool was_frozen = frozen();
  frozen_allocate_count = std::min((int)allocate_count, (int)tensor_groups.size());
  return !was_frozen;
}

template <core::concepts::Game Game>
NNEvaluationService<Game>::BatchDataSliceAllocator::BatchDataSliceAllocator(
  int batch_size_limit, core::PerfStats& perf_stats)
    : batch_size_limit_(batch_size_limit), perf_stats_(perf_stats) {
  add_batch_data();
}

template <core::concepts::Game Game>
NNEvaluationService<Game>::BatchDataSliceAllocator::~BatchDataSliceAllocator() {
  for (BatchData* batch_data : pending_batch_datas_) {
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
  // Attempt a lock-free allocation:
  BatchData* batch_data = current_batch_data_;  // important to do this to avoid race condition
  int candidate_pre_allocate_count =
    batch_data->allocate_count.fetch_add(n, std::memory_order_relaxed);
  int candidate_post_allocate_count = candidate_pre_allocate_count + n;

  if (candidate_post_allocate_count <= batch_data->capacity()) {
    // Nice! We can allocate entirely from the current batch data without contention.
    slices[0].batch_data = batch_data;
    slices[0].start_row = candidate_pre_allocate_count;
    slices[0].num_rows = n;
    return;
  }

  // Uh-oh, we overflowed the capacity! We need to acquire the mutex, roll back, and do this safely.
  std::unique_lock lock(main_mutex);

  // Note that capacity could have decreased due to a race condition. Importantly, however, capacity
  // can never *increase*.
  int capacity = batch_data->capacity();
  batch_data->allocate_count = capacity;  // partially roll back the fetch_add()
  util::release_assert(batch_data->reached_full_capacity());
  batch_data->freeze();

  // Start by allocating what we were able to validly allocate from the lock-free attempt
  int slice_index = 0;

  if (candidate_pre_allocate_count < capacity) {
    slices[slice_index].batch_data = batch_data;
    slices[slice_index].start_row = candidate_pre_allocate_count;
    slices[slice_index].num_rows = capacity - candidate_pre_allocate_count;
    n -= slices[slice_index].num_rows;
    util::release_assert(n > 0);
    slice_index++;
  }

  // Now allocate the rest

  // current_batch_data_ could have changed before we locked the mutex. We check here whether that
  // happened. We don't want to touch the initial current_batch_data_ because that is suspectible to
  // race conditions.
  batch_data = (batch_data == current_batch_data_) ? add_batch_data() : current_batch_data_;
  while (n) {
    BatchDataSlice& slice = slices[slice_index++];

    int m = std::min(n, batch_data->capacity() - batch_data->allocate_count);
    util::release_assert(m > 0);
    n -= m;

    slice.batch_data = batch_data;
    slice.start_row = batch_data->allocate_count;
    slice.num_rows = m;
    batch_data->allocate_count += m;

    if (batch_data->reached_full_capacity()) {
      batch_data->freeze();
      batch_data = add_batch_data();
    }
  }

  // Don't update current_batch_data_ until this point, to ensure that we won't encounter race
  // conditions
  current_batch_data_ = batch_data;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::BatchDataSliceAllocator::recycle(BatchData* batch_data) {
  batch_data->clear();
  batch_data_reserve_.push_back(batch_data);
}

template <core::concepts::Game Game>
bool NNEvaluationService<Game>::BatchDataSliceAllocator::freeze_up_to(
  core::nn_evaluation_sequence_id_t seq) {
  bool new_batch_needed = true;
  bool newly_frozen = false;
  for (BatchData* batch_data : pending_batch_datas_) {
    if (batch_data->sequence_id <= seq) {
      newly_frozen |= batch_data->freeze();  // newly frozen
    } else {
      new_batch_needed = false;
    }
  }

  if (new_batch_needed) {
    add_batch_data();
  }

  return newly_frozen;
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::BatchData*
NNEvaluationService<Game>::BatchDataSliceAllocator::get_frozen_pending_batch_data() const {
  if (!pending_batch_datas_.empty()) {
    BatchData* batch_data = pending_batch_datas_.front();
    if (batch_data->frozen()) {
      return batch_data;
    }
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
    perf_stats_.batch_datas_allocated++;
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
NNEvaluationResponse NNEvaluationService<Game>::evaluate(NNEvaluationRequest& request) {
  int n = request.num_fresh_items();
  if (n == 0) {
    return NNEvaluationResponse(0, core::kContinue);
  }

  if (mcts::kEnableServiceDebug) {
    LOG_INFO("{}{}() - size: {}", request.thread_id_whitespace(), __func__, n);
  }

  CacheMissInfo miss_infos[n];
  CacheLookupResult result(miss_infos);
  check_cache(request, result);
  update_perf_stats(result);

  if (result.misses == 0 && result.pending_hits == 0) {
    return NNEvaluationResponse(0, core::kContinue);
  }

  // Write to batches
  for (int i = 0; i < result.misses; ++i) {
    CacheMissInfo& miss_info = miss_infos[i];
    SubRequest& sub_request = request.sub_request(miss_info.shard_index);
    RequestItem& item = sub_request.get_fresh_item(miss_info.item_index);
    BatchData* batch_data = miss_info.batch_data;
    int row = miss_info.row;
    write_to_batch(item, batch_data, row);
  }

  return NNEvaluationResponse(result.max_sequence_id, core::kYield);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_for(core::nn_evaluation_sequence_id_t seq) {
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("{}({}) - last={}", __func__, seq, last_evaluated_sequence_id_);
  }

  if (last_evaluated_sequence_id_ >= seq) return;

  std::unique_lock lock(main_mutex_);
  if (batch_data_slice_allocator_.freeze_up_to(seq)) {
    cv_main_.notify_all();
  }

  cv_eval_.wait(lock, [&] { return !active() || last_evaluated_sequence_id_ >= seq; });
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("{}({}) - done waiting, last={}", __func__, seq, last_evaluated_sequence_id_);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::end_session(int num_game_threads) {
  if (session_ended_) return;

  int64_t cache_hits = perf_stats_.cache_hits;
  int64_t cache_misses = perf_stats_.cache_misses;
  int64_t cache_attempts = cache_hits + cache_misses;
  float cache_hit_pct = cache_attempts > 0 ? 100.0 * cache_hits / cache_attempts : 0.0f;

  int64_t positions_evaluated = perf_stats_.positions_evaluated;
  int64_t batches_evaluated = perf_stats_.batches_evaluated;
  int64_t full_batches_evaluated = perf_stats_.full_batches_evaluated;

  int64_t check_cache_mutex_time_ns = perf_stats_.check_cache_mutex_time_ns;
  int64_t check_cache_insert_time_ns = perf_stats_.check_cache_insert_time_ns;
  int64_t check_cache_alloc_time_ns = perf_stats_.check_cache_alloc_time_ns;
  int64_t check_cache_set_time_ns = perf_stats_.check_cache_set_time_ns;
  int64_t batch_ready_wait_time_ns = perf_stats_.batch_ready_wait_time_ns;
  int64_t gpu_copy_time_ns = perf_stats_.gpu_copy_time_ns;
  int64_t model_eval_time_ns = perf_stats_.model_eval_time_ns;

  int batch_datas_allocated = perf_stats_.batch_datas_allocated;

  float max_batch_pct =
      batches_evaluated > 0 ? 100.0 * full_batches_evaluated / batches_evaluated : 0.0f;

  float avg_batch_size =
      batches_evaluated > 0 ? positions_evaluated * 1.0 / batches_evaluated : 0.0f;

  float avg_check_cache_mutex_time_ms = 1e-6 * check_cache_mutex_time_ns / num_game_threads;
  float avg_check_cache_insert_time_ms = 1e-6 * check_cache_insert_time_ns / num_game_threads;
  float avg_check_cache_alloc_time_ms = 1e-6 * check_cache_alloc_time_ns / num_game_threads;
  float avg_check_cache_set_time_ms = 1e-6 * check_cache_set_time_ns / num_game_threads;
  float avg_batch_ready_wait_time_ms = 1e-6 * batch_ready_wait_time_ns / batches_evaluated;
  float avg_gpu_copy_time_ms = 1e-6 * gpu_copy_time_ns / batches_evaluated;
  float avg_model_eval_time_ms = 1e-6 * model_eval_time_ns / batches_evaluated;

  util::KeyValueDumper::add(dump_key("cache hits"), "%ld", cache_hits);
  util::KeyValueDumper::add(dump_key("cache misses"), "%ld", cache_misses);
  util::KeyValueDumper::add(dump_key("cache hit rate"), "%.2f%%", cache_hit_pct);
  util::KeyValueDumper::add(dump_key("evaluated positions"), "%ld", positions_evaluated);
  util::KeyValueDumper::add(dump_key("batches evaluated"), "%ld", batches_evaluated);
  util::KeyValueDumper::add(dump_key("max batch pct"), "%.2f%%", max_batch_pct);
  util::KeyValueDumper::add(dump_key("avg batch size"), "%.2f", avg_batch_size);
  util::KeyValueDumper::add(dump_key("check cache mutex time"), "%.3fms",
                            avg_check_cache_mutex_time_ms);
  util::KeyValueDumper::add(dump_key("check cache insert time"), "%.3fms",
                            avg_check_cache_insert_time_ms);
  util::KeyValueDumper::add(dump_key("check cache alloc time"), "%.3fms",
                            avg_check_cache_alloc_time_ms);
  util::KeyValueDumper::add(dump_key("check cache set time"), "%.3fms",
                            avg_check_cache_set_time_ms);
  util::KeyValueDumper::add(dump_key("avg batch ready wait time"), "%.3fms",
                            avg_batch_ready_wait_time_ms);
  util::KeyValueDumper::add(dump_key("avg gpu copy time"), "%.3fms", avg_gpu_copy_time_ms);
  util::KeyValueDumper::add(dump_key("avg model eval time"), "%.3fms", avg_model_eval_time_ms);
  util::KeyValueDumper::add(dump_key("batch datas allocated"), "%d", batch_datas_allocated);
  session_ended_ = true;
}

template <core::concepts::Game Game>
core::PerfStats NNEvaluationService<Game>::get_perf_stats() {
  std::unique_lock lock(perf_stats_mutex_);
  core::PerfStats perf_stats_copy = perf_stats_;
  new (&perf_stats_) core::PerfStats();
  lock.unlock();

  return perf_stats_copy;
}

template <core::concepts::Game Game>
std::string NNEvaluationService<Game>::dump_key(const char* descr) {
  return util::create_string("NN-%d %s", instance_id_, descr);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::check_cache(NNEvaluationRequest& request,
                                            CacheLookupResult& result) {
  core::cache_shard_index_t shard_index = 0;
  for (; shard_index < mcts::kNumCacheShards; ++shard_index) {
    SubRequest& sub_request = request.sub_request(shard_index);

    int num_fresh_items = sub_request.num_fresh_items();
    if (num_fresh_items == 0) continue;

    int current_miss_count = result.misses;
    Shard& shard = shards_[shard_index];

    core::PerfStatsClocker clocker1(result.check_cache_mutex_time_ns);
    std::unique_lock shard_lock(shard.mutex);
    core::PerfStatsClocker clocker2(clocker1, result.check_cache_insert_time_ns);

    // Lazily decrement old reference counts. We do this here to ensure that all reference-count
    // changes to all NNEvaluation objects are done under the same lock. If we did not do it like
    // this, then we would need to use various synchronization primitives like
    // std::atomic<std::shared_ptr> for thread-safety, which would add significant overhead.
    for (RequestItem& item : sub_request.stale_items()) {
      decrement_ref_count(shard, item.eval());
    }
    sub_request.clear_stale_items();

    for (int item_index = 0; item_index < num_fresh_items; ++item_index) {
      RequestItem& item = sub_request.get_fresh_item(item_index);
      util::release_assert(item.eval() == nullptr);

      bool hit_cache = true;
      const CacheKey& cache_key = item.cache_key();
      auto value_creator = [&]() {
        hit_cache = false;
        return shard.eval_pool.alloc();
      };

      NNEvaluation* eval = shard.eval_cache.insert_if_missing(cache_key, value_creator);
      eval->increment_ref_count();  // ref_count++ because item is now holding it
      item.set_eval(eval);
      if (hit_cache) {
        util::release_assert(eval->sequence_id() > 0);
        if (eval->pending()) {
          result.max_sequence_id = std::max(result.max_sequence_id, eval->sequence_id());
          result.pending_hits++;
        } else {
          result.non_pending_hits++;
        }
      } else {
        util::release_assert(eval->sequence_id() == 0);
        eval->increment_ref_count();  // ref_count++ because cache is now holding it
        result.miss_infos[result.misses].item_index = item_index;
        result.miss_infos[result.misses].shard_index = shard_index;
        result.misses++;
      }
    }

    if (result.misses > current_miss_count) {
      core::PerfStatsClocker clocker3(clocker2, result.check_cache_alloc_time_ns);
      // Assign the missed items to BatchData's.
      //
      // Don't actually write to the BatchData's yet, since we don't need to do that under the
      // cache_mutex_ lock.

      // In the pathological worst case, each BatchData only gets allocated 1 row, hence we size
      // the slices array to the number of misses.
      int n_misses_for_shard = result.misses - current_miss_count;
      BatchDataSlice slices[n_misses_for_shard];
      batch_data_slice_allocator_.allocate_slices(slices, n_misses_for_shard, main_mutex_);

      core::PerfStatsClocker clocker4(clocker3, result.check_cache_set_time_ns);

      int slice_index = 0;
      int slice_offset = 0;

      for (int i = current_miss_count; i < result.misses; ++i) {
        CacheMissInfo& miss_info = result.miss_infos[i];
        RequestItem& item = sub_request.get_fresh_item(miss_info.item_index);

        BatchDataSlice& slice = slices[slice_index];
        BatchData* batch_data = slice.batch_data;
        miss_info.batch_data = batch_data;
        miss_info.row = slice.start_row + slice_offset;

        item.eval()->set_sequence_id(slice.batch_data->sequence_id);

        result.max_sequence_id = std::max(result.max_sequence_id, slice.batch_data->sequence_id);
        slice_offset++;
        if (slice_offset == slices[slice_index].num_rows) {
          slice_index++;
          slice_offset = 0;
        }
      }
    }
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::decrement_ref_count(Shard& shard, NNEvaluation* eval) {
  if (eval->decrement_ref_count()) {
    shard.eval_pool.free(eval);
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

  batch_data->write_count++;
  if (batch_data->reached_full_capacity()) {
    cv_main_.notify_all();
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::update_perf_stats(const CacheLookupResult& result) {
  std::unique_lock<std::mutex> perf_stats_lock(perf_stats_mutex_);
  perf_stats_.cache_misses += result.misses;
  perf_stats_.cache_hits += result.non_pending_hits + result.pending_hits;
  perf_stats_.check_cache_mutex_time_ns += result.check_cache_mutex_time_ns;
  perf_stats_.check_cache_insert_time_ns += result.check_cache_insert_time_ns;
  perf_stats_.check_cache_alloc_time_ns += result.check_cache_alloc_time_ns;
  perf_stats_.check_cache_set_time_ns += result.check_cache_set_time_ns;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::update_perf_stats(int num_rows) {
  std::lock_guard guard(perf_stats_mutex_);
  perf_stats_.positions_evaluated += num_rows;
  perf_stats_.batches_evaluated++;
  bool full = num_rows == params_.batch_size_limit;
  if (full) perf_stats_.full_batches_evaluated++;
}


template <core::concepts::Game Game>
void NNEvaluationService<Game>::loop() {
  while (active()) {
    set_deadline();
    load_initial_weights_if_necessary();
    wait_for_unpause();
    wait_until_batch_ready();
    batch_evaluate();
    profiler_.dump(64);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::set_deadline() {
  deadline_ = std::chrono::steady_clock::now() + timeout_duration_;
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
void NNEvaluationService<Game>::wait_until_batch_ready() {
  profiler_.record(NNEvaluationServiceRegion::kWaitingUntilBatchReady);

  std::unique_lock lock(main_mutex_);
  core::PerfStatsClocker clocker(perf_stats_.batch_ready_wait_time_ns);

  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<---------------------- {}::{}() ---------------------->", cls, func);
  }

  auto predicate = [&] {
    if (!active() || paused_) return true;
    BatchData* batch_data = batch_data_slice_allocator_.get_frozen_pending_batch_data();
    if (batch_data) {
      if (mcts::kEnableServiceDebug) {
        LOG_INFO("<---------------------- {}::{}() (count:{}) ---------------------->",
                 cls, func, (int)batch_data->allocate_count);
      }
      return true;
    }
    if (mcts::kEnableServiceDebug) {
      LOG_INFO("<---------------------- {}::{}() still waiting ---------------------->", cls, func);
    }
    return false;
  };

  cv_main_.wait_until(lock, deadline_, predicate);

  if (!active() || paused_) return;
  BatchData* batch_data = batch_data_slice_allocator_.get_frozen_pending_batch_data();
  if (!batch_data) {
    // This means that we timed out, but the current batch is not yet frozen. We need to wait
    // further until it is frozen.
    cv_main_.wait(lock, predicate);
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::batch_evaluate() {
  if (!active() || paused_) return;

  std::unique_lock lock(main_mutex_);
  BatchData* batch_data = batch_data_slice_allocator_.get_frozen_pending_batch_data();
  if (!batch_data) return;
  lock.unlock();

  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableServiceDebug) {
    LOG_INFO("<---------------------- {}::{}() (seq:{}, count:{}) ---------------------->",
             cls, func, batch_data->sequence_id, (int)batch_data->allocate_count);
  }

  profiler_.record(NNEvaluationServiceRegion::kCopyingCpuToGpu);
  core::PerfStatsClocker gpu_copy_clocker(perf_stats_.gpu_copy_time_ns);
  int num_rows = batch_data->write_count;
  batch_data->copy_input_to(num_rows, full_input_);
  auto input_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                 eigen_util::to_int64_std_array_v<InputShape>);
  torch::Tensor full_input_torch = torch::from_blob(full_input_.data(), input_shape);
  torch_input_gpu_.copy_(full_input_torch);

  profiler_.record(NNEvaluationServiceRegion::kEvaluatingNeuralNet);
  core::PerfStatsClocker model_eval_clocker(gpu_copy_clocker, perf_stats_.model_eval_time_ns);
  net_.predict(input_vec_, torch_policy_, torch_value_, torch_action_value_);

  profiler_.record(NNEvaluationServiceRegion::kCopyingToPool);
  core::PerfStatsClocker gpu_copy_clocker2(model_eval_clocker, perf_stats_.gpu_copy_time_ns);
  for (int i = 0; i < num_rows; ++i) {
    TensorGroup& group = batch_data->tensor_groups[i];
    group.load_output_from(i, torch_policy_, torch_value_, torch_action_value_);
    group.eval->init(group.value, group.policy, group.action_values, group.valid_actions, group.sym,
                     group.active_seat, group.action_mode);
  }
  gpu_copy_clocker2.stop();

  lock.lock();
  util::release_assert(last_evaluated_sequence_id_ < batch_data->sequence_id);
  last_evaluated_sequence_id_ = batch_data->sequence_id;
  batch_data_slice_allocator_.recycle(batch_data);
  lock.unlock();
  cv_eval_.notify_all();

  update_perf_stats(num_rows);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::reload_weights(const std::vector<char>& buf,
                                               const std::string& cuda_device) {
  LOG_INFO("NNEvaluationService: reloading network weights...");
  util::release_assert(paused_, "%s() called while not paused", __func__);

  std::ispanstream stream{std::span<const char>(buf)};
  std::unique_lock net_weights_lock(net_weights_mutex_);
  net_.load_weights(stream, cuda_device);
  initial_weights_loaded_ = true;
  net_weights_lock.unlock();
  cv_net_weights_.notify_all();

  LOG_INFO("NNEvaluationService: clearing network cache...");
  // TODO: the below can be done in parallel. Consider initializing a set of threads that do this
  // in the constructor, and simply waking them up here.
  for (Shard& shard : shards_) {
    std::unique_lock shard_lock(shard.mutex);
    shard.eval_cache.clear();
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
  LOG_INFO("NNEvaluationService: unpause complete!");
}

}  // namespace mcts
