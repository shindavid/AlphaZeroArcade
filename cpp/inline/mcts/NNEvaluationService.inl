#include <mcts/NNEvaluationService.hpp>

#include <util/Asserts.hpp>

#include <boost/json/src.hpp>

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
    if (thread_->joinable()) thread_->detach();
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
inline NNEvaluationService<Game>::NNEvaluationService(
    const NNEvaluationServiceParams& params)
    : instance_id_(instance_count_++),
      params_(params),
      batch_data_(params.batch_size_limit),
      full_input_(util::to_std_array<int64_t>(params.batch_size_limit,
                                              eigen_util::to_int64_std_array_v<InputShape>)),
      cache_(params.cache_size),
      timeout_duration_(params.nn_eval_timeout_ns) {
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

  torch_input_gpu_ = torch::empty(input_shape, torch_util::to_dtype_v<float>)
                         .to(at::Device(params.cuda_device));
  torch_policy_ = torch::empty(policy_shape, torch_util::to_dtype_v<float>);
  torch_value_ = torch::empty(value_shape, torch_util::to_dtype_v<float>);
  torch_action_value_ = torch::empty(policy_shape, torch_util::to_dtype_v<float>);

  input_vec_.push_back(torch_input_gpu_);
  deadline_ = std::chrono::steady_clock::now();

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
inline void NNEvaluationService<Game>::tensor_group_t::load_output_from(
    int row, torch::Tensor& torch_policy, torch::Tensor& torch_value,
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
inline NNEvaluationService<Game>::batch_data_t::batch_data_t(int batch_size) {
  tensor_groups_ = new tensor_group_t[batch_size];
}

template <core::concepts::Game Game>
inline NNEvaluationService<Game>::batch_data_t::~batch_data_t() {
  delete[] tensor_groups_;
}

template <core::concepts::Game Game>
inline void NNEvaluationService<Game>::batch_data_t::copy_input_to(
    int num_rows, DynamicInputTensor& full_input) {
  float* full_input_data = full_input.data();
  constexpr size_t input_size = InputShape::total_size;
  int r = 0;
  for (int row = 0; row < num_rows; row++) {
    const tensor_group_t& group = tensor_groups_[row];
    memcpy(full_input_data + r, group.input.data(), input_size * sizeof(float));
    r += input_size;
  }
}

template <core::concepts::Game Game>
inline NNEvaluationService<Game>::~NNEvaluationService() {
  disconnect();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::evaluate(const NNEvaluationRequest& request) {
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "evaluate() - size: " << request.items().size();
  }

  while (true) {
    int my_claim_count = 0;
    int other_claim_count = 0;

    std::unique_lock cache_lock(cache_mutex_);
    cv_cache_.wait(cache_lock, [&] {
      check_cache(request, my_claim_count, other_claim_count);
      return my_claim_count > 0 || other_claim_count == 0;
    });
    cache_lock.unlock();
    cv_cache_.notify_all();

    if (!my_claim_count) break;

    std::unique_lock<std::mutex> metadata_lock(batch_metadata_.mutex);
    wait_until_batch_reservable(request, metadata_lock);
    index_reservation_t reservation = make_reservation(request, my_claim_count, metadata_lock);
    metadata_lock.unlock();

    int reservation_count = 0;
    for (const RequestItem& item : request.items()) {
      if (item.eval_state() != NNEvaluationRequest::kClaimedByMe) continue;
      int reserve_index = reservation.start_index + reservation_count;
      tensorize_and_transform_input(request, item, reserve_index);
      reservation_count++;
      if (reservation_count == reservation.num_indices) break;
    }

    metadata_lock.lock();
    increment_commit_count(request, reservation.num_indices);
    wait_for_eval(request, metadata_lock);
    wait_until_all_read(request, reservation.num_indices, metadata_lock);
    metadata_lock.unlock();

    cv_evaluate_.notify_all();

    if (mcts::kEnableDebug) {
      LOG_INFO << request.thread_id_whitespace() << "  evaluated!";
    }
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::end_session() {
  if (session_ended_) return;

  int64_t cache_hits = perf_stats_.cache_hits;
  int64_t cache_misses = perf_stats_.cache_misses;
  int64_t cache_attempts = cache_hits + cache_misses;
  float cache_hit_pct = cache_attempts > 0 ? 100.0 * cache_hits / cache_attempts : 0.0f;

  int64_t positions_evaluated = perf_stats_.positions_evaluated;
  int64_t batches_evaluated = perf_stats_.batches_evaluated;
  int64_t full_batches_evaluated = perf_stats_.full_batches_evaluated;

  float max_batch_pct =
      batches_evaluated > 0 ? 100.0 * full_batches_evaluated / batches_evaluated : 0.0f;

  float avg_batch_size =
      batches_evaluated > 0 ? positions_evaluated * 1.0 / batches_evaluated : 0.0f;

  util::KeyValueDumper::add(dump_key("cache hits"), "%ld", cache_hits);
  util::KeyValueDumper::add(dump_key("cache misses"), "%ld", cache_misses);
  util::KeyValueDumper::add(dump_key("cache hit rate"), "%.2f%%", cache_hit_pct);
  util::KeyValueDumper::add(dump_key("evaluated positions"), "%ld", positions_evaluated);
  util::KeyValueDumper::add(dump_key("batches evaluated"), "%ld", batches_evaluated);
  util::KeyValueDumper::add(dump_key("max batch pct"), "%.2f%%", max_batch_pct);
  util::KeyValueDumper::add(dump_key("avg batch size"), "%.2f", avg_batch_size);
  session_ended_ = true;
}

template <core::concepts::Game Game>
core::perf_stats_t NNEvaluationService<Game>::get_perf_stats() {
  std::unique_lock lock(perf_stats_mutex_);
  core::perf_stats_t perf_stats_copy = perf_stats_;
  new (&perf_stats_) core::perf_stats_t();
  lock.unlock();

  return perf_stats_copy;
}

template <core::concepts::Game Game>
std::string NNEvaluationService<Game>::dump_key(const char* descr) {
  return util::create_string("NN-%d %s", instance_id_, descr);
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::batch_evaluate() {
  std::unique_lock batch_metadata_lock(batch_metadata_.mutex);
  std::unique_lock batch_data_lock(batch_data_.mutex);

  util::debug_assert(batch_metadata_.reserve_index > 0);
  util::debug_assert(batch_metadata_.reserve_index == batch_metadata_.commit_count);

  if (mcts::kEnableDebug) {
    LOG_INFO << "<---------------------- NNEvaluationService::"
             << __func__ << "(" << batch_metadata_.repr() << ") ---------------------->";
  }

  profiler_.record(NNEvaluationServiceRegion::kCopyingCpuToGpu);
  int num_rows = batch_metadata_.reserve_index;
  batch_data_.copy_input_to(num_rows, full_input_);
  auto input_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                 eigen_util::to_int64_std_array_v<InputShape>);
  torch::Tensor full_input_torch = torch::from_blob(full_input_.data(), input_shape);
  torch_input_gpu_.copy_(full_input_torch);

  profiler_.record(NNEvaluationServiceRegion::kEvaluatingNeuralNet);
  net_.predict(input_vec_, torch_policy_, torch_value_, torch_action_value_);

  profiler_.record(NNEvaluationServiceRegion::kCopyingToPool);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    tensor_group_t& group = batch_data_.tensor_groups_[i];
    group.load_output_from(i, torch_policy_, torch_value_, torch_action_value_);
    eval_ptr_data_t& edata = group.eval_ptr_data;

    eigen_util::right_rotate(eigen_util::reinterpret_as_array(group.value), group.current_player);

    group::element_t inv_sym = Game::SymmetryGroup::inverse(edata.sym);
    Game::Symmetries::apply(group.policy, inv_sym);
    Game::Symmetries::apply(group.action_values, inv_sym);
    edata.eval_ptr.store(std::make_shared<NNEvaluation>(group.value, group.policy,
                                                        group.action_values, edata.valid_actions));
  }

  profiler_.record(NNEvaluationServiceRegion::kAcquiringCacheMutex);
  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  profiler_.record(NNEvaluationServiceRegion::kFinishingUp);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    const eval_ptr_data_t& edata = batch_data_.tensor_groups_[i].eval_ptr_data;
    cache_.insert(edata.cache_key, edata.eval_ptr);
  }
  cache_lock.unlock();
  cv_cache_.notify_all();

  int batch_size = batch_metadata_.reserve_index;
  bool full = batch_size == params_.batch_size_limit;

  {
    std::lock_guard guard(perf_stats_mutex_);
    perf_stats_.positions_evaluated += batch_size;
    perf_stats_.batches_evaluated++;
    if (full) perf_stats_.full_batches_evaluated++;
  }

  batch_metadata_.unread_count = batch_metadata_.commit_count;
  batch_metadata_.reserve_index = 0;
  batch_metadata_.commit_count = 0;
  batch_metadata_.accepting_reservations = true;
  cv_evaluate_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::loop() {
  while (active()) {
    load_initial_weights_if_necessary();
    wait_for_unpause();
    wait_until_batch_ready();
    if (wait_for_first_reservation()) continue;
    wait_for_last_reservation();
    wait_for_commits();
    batch_evaluate();
    profiler_.dump();
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::check_cache(const NNEvaluationRequest& request, int& my_claim_count,
                                            int& other_claim_count) {
  request.thread_profiler()->record(SearchThreadRegion::kCheckingCache);

  my_claim_count = 0;
  other_claim_count = 0;
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  waiting for cache lock (my_claim_count="
             << my_claim_count << ", other_claim_count=" << other_claim_count << ")";
  }

  for (RequestItem& item : request.items()) {
    if (item.eval()) continue;
    auto lookup = cache_.get(item.cache_key());
    if (lookup.has_value()) {
      if (lookup.value()) {
        item.set_eval(lookup.value());
        item.set_eval_state(NNEvaluationRequest::kCompleted);
      } else if (item.eval_state() == NNEvaluationRequest::kClaimedByMe) {
        my_claim_count++;
      } else {
        item.set_eval_state(NNEvaluationRequest::kClaimedByOther);
        other_claim_count++;
      }
    } else {
      cache_.insert(item.cache_key(), NNEvaluation_asptr());
      item.set_eval_state(NNEvaluationRequest::kClaimedByMe);
      my_claim_count++;
    }
  }
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_until_batch_reservable(
    const NNEvaluationRequest& request, std::unique_lock<std::mutex>& metadata_lock) {
  request.thread_profiler()->record(SearchThreadRegion::kWaitingUntilBatchReservable);

  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
             << ")...";
  }
  cv_evaluate_.wait(metadata_lock, [&] {
    if (batch_metadata_.unread_count == 0 &&
        batch_metadata_.reserve_index < params_.batch_size_limit &&
        batch_metadata_.accepting_reservations)
      return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
               << ") still waiting...";
    }
    return false;
  });
}

template <core::concepts::Game Game>
typename NNEvaluationService<Game>::index_reservation_t
NNEvaluationService<Game>::make_reservation(const NNEvaluationRequest& request, int count,
                                            std::unique_lock<std::mutex>& metadata_lock) {
  request.thread_profiler()->record(SearchThreadRegion::kMisc);

  int start_index = batch_metadata_.reserve_index;
  int num_indices = std::min(count, params_.batch_size_limit - start_index);
  util::debug_assert(num_indices > 0);
  batch_metadata_.reserve_index += num_indices;
  if (start_index == 0) {
    deadline_ = std::chrono::steady_clock::now() + timeout_duration_;
  }
  util::debug_assert(batch_metadata_.commit_count < batch_metadata_.reserve_index);

  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
             << ") allocation complete (count=" << count << ", start_index=" << start_index << ")";
  }
  cv_service_loop_.notify_one();

  /*
   * At this point, the work units are effectively RESERVED but not COMMITTED.
   *
   * The significance of being reserved is that other search threads will be blocked from reserving
   * if the batch is fully reserved.
   *
   * The significance of not yet being committed is that the service thread won't yet proceed with
   * model eval.
   */
  return index_reservation_t{start_index, num_indices};
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::tensorize_and_transform_input(const NNEvaluationRequest& request,
                                                              const RequestItem& item,
                                                              int reserve_index) {
  request.thread_profiler()->record(SearchThreadRegion::kTensorizing);

  const cache_key_t& cache_key = item.cache_key();

  const auto& stable_data = item.node()->stable_data();
  const ActionMask& valid_action_mask = stable_data.valid_action_mask;
  core::seat_index_t current_player = stable_data.current_player;
  group::element_t sym = std::get<1>(cache_key);
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

  std::unique_lock<std::mutex> lock(batch_data_.mutex);

  tensor_group_t& group = batch_data_.tensor_groups_[reserve_index];
  group.input = input;
  group.current_player = current_player;
  group.eval_ptr_data.eval_ptr.store(nullptr);
  group.eval_ptr_data.cache_key = cache_key;
  group.eval_ptr_data.valid_actions = valid_action_mask;
  group.eval_ptr_data.sym = sym;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::increment_commit_count(const NNEvaluationRequest& request,
                                                       int count) {
  request.thread_profiler()->record(SearchThreadRegion::kIncrementingCommitCount);

  batch_metadata_.commit_count += count;
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  " << __func__ << "("
             << batch_metadata_.repr() << ")...";
  }
  cv_service_loop_.notify_one();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_for_eval(const NNEvaluationRequest& request,
                                              std::unique_lock<std::mutex>& metadata_lock) {
  const char* func = __func__;
  request.thread_profiler()->record(SearchThreadRegion::kWaitingForReservationProcessing);
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
             << ")...";
  }
  cv_evaluate_.wait(metadata_lock, [&] {
    if (batch_metadata_.reserve_index == 0) return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
               << ") still waiting...";
    }
    return false;
  });

  // return batch_data_.tensor_groups_[reserve_index].eval_ptr_data.eval_ptr.load();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_until_all_read(
    const NNEvaluationRequest& request, int count, std::unique_lock<std::mutex>& metadata_lock) {
  util::debug_assert(batch_metadata_.unread_count >= count);
  batch_metadata_.unread_count -= count;
  cv_service_loop_.notify_all();

  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
             << ")...";
  }
  cv_evaluate_.wait(metadata_lock, [&] {
    if (batch_metadata_.unread_count == 0) return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
               << ") still waiting...";
    }
    return false;
  });
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_for_unpause() {
  if (!skip_next_pause_receipt_ && !paused_) return;  // early exit for common case, bypassing lock

  LOG_INFO << "NNEvaluationService: wait_for_unpause - acquiring pause_mutex_";
  std::unique_lock lock(pause_mutex_);
  if (skip_next_pause_receipt_) {
    LOG_INFO << "NNEvaluationService: skipping handle_pause_receipt";
    skip_next_pause_receipt_ = false;
  } else {
    net_.deactivate();
    LOG_INFO << "NNEvaluationService: handle_pause_receipt";
    core::LoopControllerClient::get()->handle_pause_receipt();
  }
  cv_paused_.wait(lock, [&] { return !paused_; });
  lock.unlock();

  LOG_INFO << "NNEvaluationService: handle_unpause_receipt";
  core::LoopControllerClient::get()->handle_unpause_receipt();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::load_initial_weights_if_necessary() {
  if (initial_weights_loaded_) return;

  LOG_INFO << "NNEvaluationService: requesting weights...";

  core::LoopControllerClient::get()->request_weights();
  std::unique_lock<std::mutex> net_weights_lock(net_weights_mutex_);
  cv_net_weights_.wait(net_weights_lock, [&] { return initial_weights_loaded_; });
  LOG_INFO << "NNEvaluationService: weights loaded!";
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::reload_weights(std::stringstream& ss,
                                               const std::string& cuda_device) {
  LOG_INFO << "NNEvaluationService: reloading network weights...";
  util::release_assert(paused_, "%s() called while not paused", __func__);
  std::unique_lock net_weights_lock(net_weights_mutex_);
  net_.load_weights(ss, cuda_device);
  initial_weights_loaded_ = true;
  net_weights_lock.unlock();
  cv_net_weights_.notify_all();

  LOG_INFO << "NNEvaluationService: clearing network cache...";
  std::unique_lock cache_lock(cache_mutex_);
  cache_.clear();
  cache_lock.unlock();

  cv_cache_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::pause() {
  LOG_INFO << "NNEvaluationService: pausing";
  std::unique_lock lock(pause_mutex_);
  if (paused_) {
    net_.deactivate();
    LOG_INFO << "NNEvaluationService: handle_pause_receipt (already paused)";
    core::LoopControllerClient::get()->handle_pause_receipt();
    return;
  }
  paused_ = true;

  if (!initial_weights_loaded_) {
    net_.deactivate();
    skip_next_pause_receipt_ = true;
    LOG_INFO << "NNEvaluationService: handle_pause_receipt (skip next)";
    core::LoopControllerClient::get()->handle_pause_receipt();
  }
  LOG_INFO << "NNEvaluationService: pause complete!";

  lock.unlock();
  cv_paused_.notify_all();
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::unpause() {
  LOG_INFO << "NNEvaluationService: unpausing";
  std::unique_lock lock(pause_mutex_);
  net_.activate();
  if (!paused_) {
    LOG_INFO << "NNEvaluationService: handle_unpause_receipt (already unpaused)";
    core::LoopControllerClient::get()->handle_unpause_receipt();
    return;
  }
  paused_ = false;
  lock.unlock();
  cv_paused_.notify_all();
  LOG_INFO << "NNEvaluationService: unpause complete!";
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_until_batch_ready() {
  profiler_.record(NNEvaluationServiceRegion::kWaitingUntilBatchReady);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
             << ") ---------------------->";
  }
  cv_service_loop_.wait(lock, [&] {
    if (batch_metadata_.unread_count == 0) return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
               << ") still waiting ---------------------->";
    }
    return false;
  });
}

template <core::concepts::Game Game>
bool NNEvaluationService<Game>::wait_for_first_reservation() {
  profiler_.record(NNEvaluationServiceRegion::kWaitingForFirstReservation);
  auto now = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
             << ") ---------------------->";
  }

  // Without the deadline, we can end up deadlocking here if the loop-controller issues a pause
  // command while we're waiting for the first reservation.
  auto deadline = now + std::chrono::milliseconds(100);
  cv_service_loop_.wait_until(lock, deadline, [&] {
    if (batch_metadata_.reserve_index > 0) return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
               << ") still waiting ---------------------->";
    }
    return false;
  });

  return batch_metadata_.reserve_index == 0;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_for_last_reservation() {
  profiler_.record(NNEvaluationServiceRegion::kWaitingForLastReservation);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
             << ") ---------------------->";
  }
  cv_service_loop_.wait_until(lock, deadline_, [&] {
    if (batch_metadata_.reserve_index == params_.batch_size_limit) return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
               << ") still waiting ---------------------->";
    }
    return false;
  });
  batch_metadata_.accepting_reservations = false;
}

template <core::concepts::Game Game>
void NNEvaluationService<Game>::wait_for_commits() {
  profiler_.record(NNEvaluationServiceRegion::kWaitingForCommits);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
             << ") ---------------------->";
  }
  cv_service_loop_.wait(lock, [&] {
    if (batch_metadata_.reserve_index == batch_metadata_.commit_count) return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
               << ") still waiting ---------------------->";
    }
    return false;
  });
}

}  // namespace mcts
