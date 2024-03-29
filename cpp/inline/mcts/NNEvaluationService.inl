#include <mcts/NNEvaluationService.hpp>

#include <util/Asserts.hpp>

#include <boost/json/src.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename NNEvaluationService<GameState, Tensorizor>::instance_map_t
    NNEvaluationService<GameState, Tensorizor>::instance_map_;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
int NNEvaluationService<GameState, Tensorizor>::instance_count_ = 0;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
NNEvaluationService<GameState, Tensorizor>* NNEvaluationService<GameState, Tensorizor>::create(
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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::connect() {
  std::lock_guard<std::mutex> guard(connection_mutex_);
  num_connections_++;
  if (thread_) return;

  load_initial_weights_if_necessary();
  thread_ = new std::thread([&] { this->loop(); });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::disconnect() {
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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void NNEvaluationService<GameState, Tensorizor>::set_profiling_dir(
    const boost::filesystem::path& profiling_dir) {
  std::string name = util::create_string("eval-%d", instance_id_);
  auto profiling_file_path = profiling_dir / util::create_string("%s.txt", name.c_str());
  profiler_.initialize_file(profiling_file_path);
  profiler_.set_name(name);
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline NNEvaluationService<GameState, Tensorizor>::NNEvaluationService(
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
  }
  auto input_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                 eigen_util::to_int64_std_array_v<InputShape>);
  auto policy_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                  eigen_util::to_int64_std_array_v<PolicyShape>);
  auto value_shape = util::to_std_array<int64_t>(params_.batch_size_limit,
                                                 eigen_util::to_int64_std_array_v<ValueShape>);

  torch_input_gpu_ = torch::empty(input_shape, torch_util::to_dtype_v<dtype>)
                         .to(at::Device(params.cuda_device));
  torch_policy_ = torch::empty(policy_shape, torch_util::to_dtype_v<PolicyScalar>);
  torch_value_ = torch::empty(value_shape, torch_util::to_dtype_v<ValueScalar>);

  input_vec_.push_back(torch_input_gpu_);
  deadline_ = std::chrono::steady_clock::now();

  if (core::LoopControllerClient::initialized()) {
    if (core::LoopControllerClient::get()->paused()) {
      this->paused_ = true;
    }
    core::LoopControllerClient::get()->add_listener(this);
  } else {
    if (!net_.loaded()) {
      throw util::CleanException(
          "MCTS player configured without --model-filename/-m and without "
          "--no-model, but --loop-controller-* options not specified");
    }
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void NNEvaluationService<GameState, Tensorizor>::tensor_group_t::load_output_from(
    int row, torch::Tensor& torch_policy, torch::Tensor& torch_value) {
  constexpr size_t policy_size = PolicyShape::total_size;
  constexpr size_t value_size = ValueShape::total_size;

  memcpy(policy.data(), torch_policy.data_ptr<PolicyScalar>() + row * policy_size,
         policy_size * sizeof(PolicyScalar));
  memcpy(value.data(), torch_value.data_ptr<ValueScalar>() + row * value_size,
         value_size * sizeof(ValueScalar));
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline NNEvaluationService<GameState, Tensorizor>::batch_data_t::batch_data_t(int batch_size) {
  tensor_groups_ = new tensor_group_t[batch_size];
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline NNEvaluationService<GameState, Tensorizor>::batch_data_t::~batch_data_t() {
  delete[] tensor_groups_;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void NNEvaluationService<GameState, Tensorizor>::batch_data_t::copy_input_to(
    int num_rows, DynamicInputFloatTensor& full_input) {
  dtype* full_input_data = full_input.data();
  constexpr size_t input_size = InputShape::total_size;
  int r = 0;
  for (int row = 0; row < num_rows; row++) {
    const tensor_group_t& group = tensor_groups_[row];
    InputFloatTensor float_input = group.input.template cast<dtype>();
    memcpy(full_input_data + r, float_input.data(), input_size * sizeof(dtype));
    r += input_size;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline NNEvaluationService<GameState, Tensorizor>::~NNEvaluationService() {
  disconnect();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename NNEvaluationService<GameState, Tensorizor>::Response
NNEvaluationService<GameState, Tensorizor>::evaluate(const Request& request) {
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "evaluate()";
  }

  cache_key_t cache_key(*request.state, request.sym_index);
  Response response = check_cache(request, cache_key);
  if (response.used_cache) return response;

  std::unique_lock<std::mutex> metadata_lock(batch_metadata_.mutex);
  wait_until_batch_reservable(request, metadata_lock);
  int my_index = allocate_reserve_index(request, metadata_lock);
  metadata_lock.unlock();

  tensorize_and_transform_input(request, cache_key, my_index);

  metadata_lock.lock();
  increment_commit_count(request);
  NNEvaluation_sptr eval_ptr = get_eval(request, my_index, metadata_lock);
  wait_until_all_read(request, metadata_lock);
  metadata_lock.unlock();

  cv_evaluate_.notify_all();

  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  evaluated!";
  }

  return Response{eval_ptr, false};
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::end_session() {
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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
core::perf_stats_t NNEvaluationService<GameState, Tensorizor>::get_perf_stats() {
  std::unique_lock lock(perf_stats_mutex_);
  core::perf_stats_t perf_stats_copy = perf_stats_;
  new (&perf_stats_) core::perf_stats_t();
  lock.unlock();

  return perf_stats_copy;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
std::string NNEvaluationService<GameState, Tensorizor>::dump_key(const char* descr) {
  return util::create_string("NN-%d %s", instance_id_, descr);
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::batch_evaluate() {
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
  net_.predict(input_vec_, torch_policy_, torch_value_);

  profiler_.record(NNEvaluationServiceRegion::kCopyingToPool);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    tensor_group_t& group = batch_data_.tensor_groups_[i];
    group.load_output_from(i, torch_policy_, torch_value_);
    eval_ptr_data_t& edata = group.eval_ptr_data;

    eigen_util::right_rotate(eigen_util::reinterpret_as_array(group.value), group.current_player);
    edata.policy_transform->undo(group.policy);
    edata.eval_ptr.store(
        std::make_shared<NNEvaluation>(group.value, group.policy, edata.valid_actions));
  }

  profiler_.record(NNEvaluationServiceRegion::kAcquiringCacheMutex);
  std::unique_lock<std::mutex> lock(cache_mutex_);
  profiler_.record(NNEvaluationServiceRegion::kFinishingUp);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    const eval_ptr_data_t& edata = batch_data_.tensor_groups_[i].eval_ptr_data;
    cache_.insert(edata.cache_key, edata.eval_ptr);
  }
  lock.unlock();

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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::loop() {
  while (active()) {
    wait_for_unpause();
    wait_until_batch_ready();
    wait_for_first_reservation();
    wait_for_last_reservation();
    wait_for_commits();
    batch_evaluate();
    profiler_.dump();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
NNEvaluationService<GameState, Tensorizor>::Response
NNEvaluationService<GameState, Tensorizor>::check_cache(const Request& request,
                                                        const cache_key_t& cache_key) {
  request.thread_profiler->record(SearchThreadRegion::kCheckingCache);

  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  waiting for cache lock...";
  }

  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  auto cached = cache_.get(cache_key);
  if (cached.has_value()) {
    if (mcts::kEnableDebug) {
      LOG_INFO << request.thread_id_whitespace() << "  hit cache";
    }
    // Technically should grab perf_stats_mutex_ here, but it's ok to be a little off
    perf_stats_.cache_hits++;
    return Response{cached.value(), true};
  }
  // Technically should grab perf_stats_mutex_ here, but it's ok to be a little off
  perf_stats_.cache_misses++;
  return Response{nullptr, false};
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_until_batch_reservable(
    const Request& request, std::unique_lock<std::mutex>& metadata_lock) {
  request.thread_profiler->record(SearchThreadRegion::kWaitingUntilBatchReservable);

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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
int NNEvaluationService<GameState, Tensorizor>::allocate_reserve_index(
    const Request& request, std::unique_lock<std::mutex>& metadata_lock) {
  request.thread_profiler->record(SearchThreadRegion::kMisc);

  int my_index = batch_metadata_.reserve_index;
  util::debug_assert(my_index < params_.batch_size_limit);
  batch_metadata_.reserve_index++;
  if (my_index == 0) {
    deadline_ = std::chrono::steady_clock::now() + timeout_duration_;
  }
  util::debug_assert(batch_metadata_.commit_count < batch_metadata_.reserve_index);

  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  " << func << "(" << batch_metadata_.repr()
             << ") allocation complete";
  }
  cv_service_loop_.notify_one();

  /*
   * At this point, the work unit is effectively RESERVED but not COMMITTED.
   *
   * The significance of being reserved is that other search threads will be blocked from reserving
   * if the batch is fully reserved.
   *
   * The significance of not yet being committed is that the service thread won't yet proceed with
   * model eval.
   */
  return my_index;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::tensorize_and_transform_input(
    const Request& request, const cache_key_t& cache_key, int reserve_index) {
  const Tensorizor& tensorizor = *request.tensorizor;
  const GameState& state = *request.state;
  const auto& stable_data = request.node->stable_data();
  const ActionMask& valid_action_mask = stable_data.valid_action_mask;
  core::seat_index_t current_player = stable_data.current_player;
  core::symmetry_index_t sym_index = cache_key.second;

  request.thread_profiler->record(SearchThreadRegion::kTensorizing);
  std::unique_lock<std::mutex> lock(batch_data_.mutex);

  tensor_group_t& group = batch_data_.tensor_groups_[reserve_index];
  tensorizor.tensorize(group.input, state);
  auto input_transform = state.template get_symmetry<InputTensor>(sym_index);
  auto policy_transform = state.template get_symmetry<PolicyTensor>(sym_index);
  input_transform->apply(group.input);

  group.current_player = current_player;
  group.eval_ptr_data.eval_ptr.store(nullptr);
  group.eval_ptr_data.cache_key = cache_key;
  group.eval_ptr_data.valid_actions = valid_action_mask;
  group.eval_ptr_data.policy_transform = policy_transform;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::increment_commit_count(const Request& request) {
  request.thread_profiler->record(SearchThreadRegion::kIncrementingCommitCount);

  batch_metadata_.commit_count++;
  if (mcts::kEnableDebug) {
    LOG_INFO << request.thread_id_whitespace() << "  " << __func__ << "("
             << batch_metadata_.repr() << ")...";
  }
  cv_service_loop_.notify_one();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename NNEvaluationService<GameState, Tensorizor>::NNEvaluation_sptr
NNEvaluationService<GameState, Tensorizor>::get_eval(const Request& request, int reserve_index,
                                                     std::unique_lock<std::mutex>& metadata_lock) {
  const char* func = __func__;
  request.thread_profiler->record(SearchThreadRegion::kWaitingForReservationProcessing);
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

  return batch_data_.tensor_groups_[reserve_index].eval_ptr_data.eval_ptr.load();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_until_all_read(
    const Request& request, std::unique_lock<std::mutex>& metadata_lock) {
  util::debug_assert(batch_metadata_.unread_count > 0);
  batch_metadata_.unread_count--;

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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_for_unpause() {
  if (!paused_) return;  // early exit for common case, bypassing lock

  core::LoopControllerClient::get()->notify_pause_received(this);

  std::unique_lock lock(pause_mutex_);
  cv_paused_.wait(lock, [&] { return !paused_; });
  lock.unlock();

  LOG_INFO << "NNEvaluationService: resuming...";
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::load_initial_weights_if_necessary() {
  LOG_INFO << "NNEvaluationService: load_initial_weights_if_necessary()";
  std::unique_lock<std::mutex> lock(net_weights_mutex_);
  if (net_.loaded()) return;

  LOG_INFO << "NNEvaluationService: requesting weights...";
  core::LoopControllerClient::get()->request_weights();
  cv_net_weights_.wait(lock, [&] { return net_.loaded(); });
  LOG_INFO << "NNEvaluationService: weights loaded!";
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::reload_weights(std::stringstream& ss) {
  LOG_INFO << "NNEvaluationService: reloading network weights...";
  util::release_assert(paused_, "%s() called while not paused", __func__);
  std::unique_lock lock1(net_weights_mutex_);
  net_.load_weights(ss, params_.cuda_device);
  cv_net_weights_.notify_all();
  lock1.unlock();

  LOG_INFO << "NNEvaluationService: clearing network cache...";
  std::unique_lock lock2(cache_mutex_);
  cache_.clear();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::pause() {
  LOG_INFO << "NNEvaluationService: pausing";
  std::unique_lock lock(pause_mutex_);
  paused_ = true;
  if (!net_.loaded()) {
    // This happens when the pause is issued during startup, when the model weights are lazily
    // loaded from the loop controller. In this case, we need to explicitly call
    // notify_pause_received() here, since the normal path to this call occurrs inside loop(),
    // which is not yet running in the lazy load case.
    //
    // Failing to call notify_pause_received() calls LoopControllerClient to get stuck inside
    // pause().
    core::LoopControllerClient::get()->notify_pause_received(this);
  }
  LOG_INFO << "NNEvaluationService: pause complete!";
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::unpause() {
  LOG_INFO << "NNEvaluationService: unpausing";
  std::unique_lock lock(pause_mutex_);
  paused_ = false;
  lock.unlock();
  cv_paused_.notify_all();
  LOG_INFO << "NNEvaluationService: unpause complete!";
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_until_batch_ready() {
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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_for_first_reservation() {
  profiler_.record(NNEvaluationServiceRegion::kWaitingForFirstReservation);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableDebug) {
    LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
             << ") ---------------------->";
  }
  cv_service_loop_.wait(lock, [&] {
    if (batch_metadata_.reserve_index > 0) return true;
    if (mcts::kEnableDebug) {
      LOG_INFO << "<---------------------- " << cls << " " << func << "(" << batch_metadata_.repr()
               << ") still waiting ---------------------->";
    }
    return false;
  });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_for_last_reservation() {
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

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_for_commits() {
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
