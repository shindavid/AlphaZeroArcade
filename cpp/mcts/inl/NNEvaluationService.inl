#include <mcts/NNEvaluationService.hpp>

#include <util/Asserts.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
int NNEvaluationService<GameState, Tensorizor>::next_instance_id_ = 0;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
bool NNEvaluationService<GameState, Tensorizor>::session_ended_ = false;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
typename NNEvaluationService<GameState, Tensorizor>::instance_map_t
    NNEvaluationService<GameState, Tensorizor>::instance_map_;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
NNEvaluationService<GameState, Tensorizor>* NNEvaluationService<GameState, Tensorizor>::create(
    const ManagerParams& manager_params) {
  boost::filesystem::path model_file_path(manager_params.model_filename);
  std::chrono::nanoseconds timeout_duration(manager_params.nn_eval_timeout_ns);

  auto it = instance_map_.find(model_file_path);
  if (it == instance_map_.end()) {
    auto instance = new NNEvaluationService(manager_params);
    instance_map_[model_file_path] = instance;
    return instance;
  }
  NNEvaluationService* instance = it->second;
  if (instance->batch_size_limit_ != manager_params.batch_size_limit) {
    throw util::Exception(
        "Conflicting NNEvaluationService::create() calls: batch_size_limit %d vs %d",
        instance->batch_size_limit_, manager_params.batch_size_limit);
  }
  if (instance->timeout_duration_ != timeout_duration) {
    throw util::Exception(
        "Conflicting NNEvaluationService::create() calls: unequal timeout_duration");
  }
  if (instance->cache_.capacity() != manager_params.cache_size) {
    throw util::Exception("Conflicting NNEvaluationService::create() calls: cache_size %ld vs %ld",
                          instance->cache_.capacity(), manager_params.cache_size);
  }
  return instance;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::connect() {
  std::lock_guard<std::mutex> guard(connection_mutex_);
  num_connections_++;
  if (thread_) return;
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
inline NNEvaluationService<GameState, Tensorizor>::NNEvaluationService(
    const ManagerParams& manager_params)
    : instance_id_(next_instance_id_++),
      net_(manager_params.model_filename, manager_params.cuda_device),
      batch_data_(manager_params.batch_size_limit),
      full_input_(util::to_std_array<int64_t>(manager_params.batch_size_limit,
                                              eigen_util::to_int64_std_array_v<InputShape>)),
      cache_(manager_params.cache_size),
      timeout_duration_(manager_params.nn_eval_timeout_ns),
      batch_size_limit_(manager_params.batch_size_limit) {
  auto input_shape =
      util::to_std_array<int64_t>(batch_size_limit_, eigen_util::to_int64_std_array_v<InputShape>);
  auto policy_shape =
      util::to_std_array<int64_t>(batch_size_limit_, eigen_util::to_int64_std_array_v<PolicyShape>);
  auto value_shape =
      util::to_std_array<int64_t>(batch_size_limit_, eigen_util::to_int64_std_array_v<ValueShape>);

  torch_input_gpu_ = torch::empty(input_shape, torch_util::to_dtype_v<dtype>)
                         .to(at::Device(manager_params.cuda_device));
  torch_policy_ = torch::empty(policy_shape, torch_util::to_dtype_v<PolicyScalar>);
  torch_value_ = torch::empty(value_shape, torch_util::to_dtype_v<ValueScalar>);

  input_vec_.push_back(torch_input_gpu_);
  deadline_ = std::chrono::steady_clock::now();

  if (kEnableProfiling) {
    std::string name = util::create_string("eval-%d", instance_id_);
    auto profiling_file_path =
        manager_params.profiling_dir() / util::create_string("%s.txt", name.c_str());
    profiler_.initialize_file(profiling_file_path);
    profiler_.set_name(name);
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
  const Node* tree = request.tree;

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("evaluate()\n");
  }

  const auto& stable_data = tree->stable_data();
  cache_key_t cache_key(stable_data.state, request.sym_index);
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
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("  evaluated!\n");
  }

  return Response{eval_ptr, false};
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::get_cache_stats(int& hits, int& misses, int& size,
                                                                 float& hash_balance_factor) const {
  hits = cache_hits_;
  misses = cache_misses_;
  size = cache_.size();
  hash_balance_factor = cache_.get_hash_balance_factor();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::record_puct_calc(bool virtual_loss_influenced) {
  this->total_puct_calcs_++;
  if (virtual_loss_influenced) {
    this->virtual_loss_influenced_puct_calcs_++;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::end_session() {
  if (session_ended_) return;

  int64_t evaluated_positions = 0;
  int64_t batches_evaluated = 0;
  for (auto it : instance_map_) {
    NNEvaluationService* service = it.second;
    evaluated_positions += service->evaluated_positions_;
    batches_evaluated += service->batches_evaluated_;
  }

  float avg_batch_size =
      batches_evaluated > 0 ? evaluated_positions * 1.0 / batches_evaluated : 0.0f;

  util::ParamDumper::add("MCTS evaluated positions", "%ld", evaluated_positions);
  util::ParamDumper::add("MCTS batches evaluated", "%ld", batches_evaluated);
  util::ParamDumper::add("MCTS avg batch size", "%.2f", avg_batch_size);
  session_ended_ = true;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
float NNEvaluationService<GameState, Tensorizor>::pct_virtual_loss_influenced_puct_calcs() {
  int64_t num = 0;
  int64_t den = 0;

  for (auto it : instance_map_) {
    NNEvaluationService* service = it.second;
    num += service->virtual_loss_influenced_puct_calcs_;
    den += service->total_puct_calcs_;
  }

  return 100.0 * num / den;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::batch_evaluate() {
  std::unique_lock batch_metadata_lock(batch_metadata_.mutex);
  std::unique_lock batch_data_lock(batch_data_.mutex);

  util::debug_assert(batch_metadata_.reserve_index > 0);
  util::debug_assert(batch_metadata_.reserve_index == batch_metadata_.commit_count);

  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- NNEvaluationService::%s(%s) ---------------------->\n",
                   __func__, batch_metadata_.repr().c_str());
  }

  profiler_.record(NNEvaluationServiceRegion::kCopyingCpuToGpu);
  int num_rows = batch_metadata_.reserve_index;
  batch_data_.copy_input_to(num_rows, full_input_);
  auto input_shape =
      util::to_std_array<int64_t>(num_rows, eigen_util::to_int64_std_array_v<InputShape>);
  torch::Tensor full_input_torch = torch::from_blob(full_input_.data(), input_shape);
  torch_input_gpu_.resize_(input_shape);
  torch_input_gpu_.copy_(full_input_torch);

  profiler_.record(NNEvaluationServiceRegion::kEvaluatingNeuralNet);
  torch_policy_.resize_(
      util::to_std_array<int64_t>(num_rows, eigen_util::to_int64_std_array_v<PolicyShape>));
  torch_value_.resize_(
      util::to_std_array<int64_t>(num_rows, eigen_util::to_int64_std_array_v<ValueShape>));
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

  evaluated_positions_ += batch_metadata_.reserve_index;
  batches_evaluated_++;

  batch_metadata_.unread_count = batch_metadata_.commit_count;
  batch_metadata_.reserve_index = 0;
  batch_metadata_.commit_count = 0;
  batch_metadata_.accepting_reservations = true;
  cv_evaluate_.notify_all();
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::loop() {
  while (active()) {
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
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("  waiting for cache lock...\n");
  }

  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  auto cached = cache_.get(cache_key);
  if (cached.has_value()) {
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(request.thread_id);
      printer.printf("  hit cache\n");
    }
    cache_hits_++;
    return Response{cached.value(), true};
  }
  cache_misses_++;
  return Response{nullptr, false};
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_until_batch_reservable(
    const Request& request, std::unique_lock<std::mutex>& metadata_lock) {
  request.thread_profiler->record(SearchThreadRegion::kWaitingUntilBatchReservable);

  const char* func = __func__;
  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&] {
    if (batch_metadata_.unread_count == 0 && batch_metadata_.reserve_index < batch_size_limit_ &&
        batch_metadata_.accepting_reservations)
      return true;
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(request.thread_id);
      printer.printf("  %s(%s) still waiting...\n", func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
int NNEvaluationService<GameState, Tensorizor>::allocate_reserve_index(
    const Request& request, std::unique_lock<std::mutex>& metadata_lock) {
  request.thread_profiler->record(SearchThreadRegion::kMisc);

  int my_index = batch_metadata_.reserve_index;
  util::debug_assert(my_index < batch_size_limit_);
  batch_metadata_.reserve_index++;
  if (my_index == 0) {
    deadline_ = std::chrono::steady_clock::now() + timeout_duration_;
  }
  util::debug_assert(batch_metadata_.commit_count < batch_metadata_.reserve_index);

  const char* func = __func__;
  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("  %s(%s) allocation complete\n", func, batch_metadata_.repr().c_str());
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
  Node* tree = request.tree;

  const auto& stable_data = tree->stable_data();
  const Tensorizor& tensorizor = stable_data.tensorizor;
  const GameState& state = stable_data.state;
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
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("  %s(%s)...\n", __func__, batch_metadata_.repr().c_str());
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
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&] {
    if (batch_metadata_.reserve_index == 0) return true;
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(request.thread_id);
      printer.printf("  %s(%s) still waiting...\n", func, batch_metadata_.repr().c_str());
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
    util::ThreadSafePrinter printer(request.thread_id);
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&] {
    if (batch_metadata_.unread_count == 0) return true;
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer(request.thread_id);
      printer.printf("  %s(%s) still waiting...\n", func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void NNEvaluationService<GameState, Tensorizor>::wait_until_batch_ready() {
  profiler_.record(NNEvaluationServiceRegion::kWaitingUntilBatchReady);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n", cls, func,
                   batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&] {
    if (batch_metadata_.unread_count == 0) return true;
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
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
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n", cls, func,
                   batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&] {
    if (batch_metadata_.reserve_index > 0) return true;
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
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
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n", cls, func,
                   batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait_until(lock, deadline_, [&] {
    if (batch_metadata_.reserve_index == batch_size_limit_) return true;
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
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
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n", cls, func,
                   batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&] {
    if (batch_metadata_.reserve_index == batch_metadata_.commit_count) return true;
    if (mcts::kEnableDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

}  // namespace mcts
