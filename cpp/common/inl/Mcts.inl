#include <common/Mcts.hpp>

#include <cmath>
#include <thread>
#include <utility>
#include <vector>

#include <EigenRand/EigenRand>

#include <util/BoostUtil.hpp>
#include <util/Config.hpp>
#include <util/EigenTorch.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts_<GameState, Tensorizor>::Params Mcts_<GameState, Tensorizor>::global_params_;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  boost::filesystem::path default_nnet_filename_path = util::Repo::root() / "c4_model.pt";
  std::string default_nnet_filename = util::Config::instance()->get(
      "nnet_filename", default_nnet_filename_path.string());

  boost::filesystem::path default_profiling_dir_path = util::Repo::root() / "output" / "mcts_profiling";
  std::string default_profiling_dir = util::Config::instance()->get(
      "mcts_profiling_dir", default_profiling_dir_path.string());

  Params& params = global_params_;
  desc.add_options()
      ("mcts-nnet-filename", po::value<std::string>(&params.nnet_filename)->default_value(default_nnet_filename),
          "nnet filename")
      ("mcts-num-search-threads", po::value<int>(&params.num_search_threads)->default_value(params.num_search_threads),
          "num search threads")
      ("mcts-batch-size-limit", po::value<int>(&params.batch_size_limit)->default_value(params.batch_size_limit),
          "batch size limit")
      ("mcts-run-offline", po::bool_switch(&params.run_offline)->default_value(false),
          "run search while opponent is thinking")
      ("mcts-offline-tree-size-limit", po::value<int>(&params.offline_tree_size_limit)->default_value(
          params.offline_tree_size_limit), "max tree size to grow to offline (only respected in --run-offline mode)")
      ("mcts-nn-eval-timeout-ns", po::value<int64_t>(&params.nn_eval_timeout_ns)->default_value(
          params.nn_eval_timeout_ns), "nn eval thread timeout in ns")
      ("mcts-cache-size", po::value<size_t>(&params.cache_size)->default_value(params.cache_size),
          "nn eval thread cache size")
      ("mcts-root-softmax-temp", po2::float_value("%.2f", &params.root_softmax_temperature), "root softmax temperature")
      ("mcts-cpuct", po2::float_value("%.2f", &params.cPUCT), "cPUCT value")
      ("mcts-dirichlet-mult", po2::float_value("%.2f", &params.dirichlet_mult), "dirichlet mult")
      ("mcts-dirichlet-alpha", po2::float_value("%.2f", &params.dirichlet_alpha), "dirichlet alpha")
      ("mcts-allow-eliminations", po::bool_switch(&params.allow_eliminations)->default_value(
          params.allow_eliminations), "allow eliminations")
#ifdef PROFILE_MCTS
      ("mcts-profiling-dir", po::value<std::string>(&params.profiling_dir)->default_value(default_profiling_dir),
          "directory in which to dump mcts profiling stats")
#endif  // PROFILE_MCTS
      ;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts_<GameState, Tensorizor>::NNEvaluationService::instance_map_t
Mcts_<GameState, Tensorizor>::NNEvaluationService::instance_map_;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::NNEvaluation::NNEvaluation(
    const ValueArray1D& value, const PolicyArray1D& policy, const ActionMask& valid_actions, float inv_temp)
{
  GameStateTypes::global_to_local(policy, valid_actions, local_policy_prob_distr_);
  value_prob_distr_ = eigen_util::softmax(value);
  local_policy_prob_distr_ = eigen_util::softmax(local_policy_prob_distr_ * inv_temp);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome, Node* parent,
    symmetry_index_t sym_index, action_index_t action, bool disable_noise)
: tensorizor_(tensorizor)
, state_(state)
, outcome_(outcome)
, valid_action_mask_(state.get_valid_actions())
, parent_(parent)
, sym_index_(sym_index)
, action_(action)
, current_player_(state.get_current_player())
, disable_noise_(disable_noise) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(const stable_data_t& data, bool prune_parent)
{
  *this = data;
  if (prune_parent) parent_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::stats_t::stats_t() {
  value_avg_.setZero();
  effective_value_avg_.setZero();
  V_floor_.setZero();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::Node(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome, symmetry_index_t sym_index,
    bool disable_noise, Node* parent, action_index_t action)
: stable_data_(tensorizor, state, outcome, parent, sym_index, action, disable_noise) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Node::Node(const Node& node, bool prune_parent)
: stable_data_(node.stable_data_, prune_parent)
, children_data_(node.children_data_)
, evaluation_(node.evaluation_)
, stats_(node.stats_) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::debug_dump() const {
  std::cout << "value[" << stats_.count_ << "]: " << stats_.value_avg_.transpose() << std::endl;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::_release(Node* protected_child) {
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    if (child != protected_child) child->_release();
  }

  if (children_data_.first_child_) delete[] children_data_.first_child_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::_adopt_children() {
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    child->stable_data_.parent_ = this;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline typename Mcts_<GameState, Tensorizor>::GlobalPolicyCountDistr
Mcts_<GameState, Tensorizor>::Node::get_effective_counts() const
{
  bool eliminated;
  {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    eliminated = stats_.eliminated_;
  }

  std::lock_guard<std::mutex> guard(children_data_mutex_);

  player_index_t cp = stable_data_.current_player_;
  GlobalPolicyCountDistr counts;
  counts.setZero();
  if (eliminated) {
    float max_V_floor = _get_max_V_floor_among_children(cp);
    for (int i = 0; i < children_data_.num_children_; ++i) {
      Node* child = children_data_.first_child_ + i;
      counts(child->action()) = (child->_V_floor(cp) == max_V_floor);
    }
    return counts;
  }
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    counts(child->action()) = child->_effective_count();
  }
  return counts;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline bool Mcts_<GameState, Tensorizor>::Node::expand_children(SearchThread* thread) {
  thread->record_for_profiling(SearchThread::kWaitingForChildrenDataMutex);
  std::lock_guard<std::mutex> guard(children_data_mutex_);

  if (_has_children()) return false;

  // TODO: use object pool
  thread->record_for_profiling(SearchThread::kAllocationChildrenMemory);
  children_data_.num_children_ = stable_data_.valid_action_mask_.count();
  void* raw_memory = operator new[](children_data_.num_children_ * sizeof(Node));
  Node* node = static_cast<Node*>(raw_memory);

  thread->record_for_profiling(SearchThread::kConstructingChildren);
  children_data_.first_child_ = node;
  for (action_index_t action : stable_data_.valid_action_mask_) {
    Tensorizor tensorizor_copy = stable_data_.tensorizor_;
    GameState state_copy = stable_data_.state_;

    // TODO: consider lazily doing these steps in visit(), since we only read them for Node's that win PUCT selection.
    symmetry_index_t sym_index = tensorizor_copy.get_random_symmetry_index(state_copy);
    GameOutcome outcome = state_copy.apply_move(action);
    tensorizor_copy.receive_state_change(state_copy, action);

    new(node++) Node(tensorizor_copy, state_copy, outcome, sym_index, true, this, action);
  }
  return true;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::backprop(const ValueProbDistr& outcome)
{
  {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    stats_.value_avg_ = (stats_.value_avg_ * stats_.count_ + outcome) / (stats_.count_ + 1);
    stats_.count_++;
    stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  }

  if (parent()) parent()->backprop(outcome);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::backprop_with_virtual_undo(
    const ValueProbDistr& value)
{
  {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    stats_.value_avg_ += (value - make_virtual_loss()) / stats_.count_;
    stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  }

  if (parent()) parent()->backprop_with_virtual_undo(value);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::virtual_backprop() {
  {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    stats_.value_avg_ = (stats_.value_avg_ * stats_.count_ + make_virtual_loss()) / (stats_.count_ + 1);
    stats_.count_++;
    stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  }
  if (parent()) parent()->virtual_backprop();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::undo_virtual_backprop() {
  {
    std::lock_guard<std::mutex> guard(stats_mutex_);
    stats_.value_avg_ = (stats_.value_avg_ * stats_.count_ - make_virtual_loss()) / std::max(1, stats_.count_ - 1);
    stats_.count_--;
    stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  }
  if (parent()) parent()->undo_virtual_backprop();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::Node::perform_eliminations(const ValueProbDistr& outcome) {
  bool recurse = false;
  {
    // TODO: tighten mutex grabs
    std::lock_guard<std::mutex> guard(stats_mutex_);

    {
      player_index_t cp = stable_data_.current_player_;
      std::lock_guard<std::mutex> guard2(children_data_mutex_);
      for (player_index_t p = 0; p < kNumPlayers; ++p) {
        if (p == cp) {
          stats_.V_floor_[p] = _get_max_V_floor_among_children(p);
        } else {
          stats_.V_floor_[p] = _get_min_V_floor_among_children(p);
        }
      }
    }

    stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
    if (_can_be_eliminated()) {
      stats_.eliminated_ = true;
      recurse = parent();
    }
  }

  if (recurse) {
    parent()->perform_eliminations(outcome);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts_<GameState, Tensorizor>::ValueArray1D
Mcts_<GameState, Tensorizor>::Node::make_virtual_loss() const {
  constexpr float x = 1.0 / (kNumPlayers - 1);
  ValueArray1D virtual_loss;
  virtual_loss = x;
  virtual_loss[stable_data_.current_player_] = 0;
  return virtual_loss;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts_<GameState, Tensorizor>::Node*
Mcts_<GameState, Tensorizor>::Node::_find_child(action_index_t action) const {
  // TODO: technically we can do a binary search here, as children should be in sorted order by action
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node *child = children_data_.first_child_ + i;
    if (child->stable_data_.action_ == action) return child;
  }
  return nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts_<GameState, Tensorizor>::Node::_get_max_V_floor_among_children(player_index_t p) const {
  float max_V_floor = 0;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    std::lock_guard<std::mutex> guard(child->stats_mutex_);
    max_V_floor = std::max(max_V_floor, child->_V_floor(p));
  }
  return max_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts_<GameState, Tensorizor>::Node::_get_min_V_floor_among_children(player_index_t p) const {
  float min_V_floor = 1;
  for (int i = 0; i < children_data_.num_children_; ++i) {
    Node* child = children_data_.first_child_ + i;
    std::lock_guard<std::mutex> guard(child->stats_mutex_);
    min_V_floor = std::min(min_V_floor, child->_V_floor(p));
  }
  return min_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::SearchThread::SearchThread(Mcts_* mcts, int thread_id)
: mcts_(mcts)
, thread_id_(thread_id) {
  auto profiling_filename = mcts->profiling_dir() / util::create_string("search%d.txt", thread_id);
  set_profiling_file(profiling_filename.c_str());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::SearchThread::~SearchThread() {
  kill();
  close_profiling_file();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::SearchThread::join() {
  if (thread_ && thread_->joinable()) thread_->join();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::SearchThread::kill() {
  join();
  if (thread_) delete thread_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::SearchThread::launch(int tree_size_limit) {
  kill();
  thread_ = new std::thread([&] { mcts_->run_search(this, tree_size_limit); });
}

/*
 * The seemingly haphazard combination of macros and runtime-branches for profiling logic is actually carefully
 * concocted! As written, we get the dual benefit of:
 *
 * 1. Zero-branching/pointer-redirection overhead in both profiling and non-profiling mode, thanks to compiler.
 * 2. Compiler checking of profiling methods even when compiled without profiling enabled.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::SearchThread::record_for_profiling(region_t region) {
  profiling_stats_t* stats = get_profiling_stats();
  if (!stats) return;  // compile-time branch

  time_point_t now = std::chrono::steady_clock::now();
  if (kEnableVerboseProfiling) {
    int64_t ns = util::ns_since_epoch(now);
    printf("%lu.%09lu s%-3d %d\n", ns / 1000000000, ns % 1000000000, thread_id_, (int) region);
  }

  if (stats->cur_region != kNumRegions) {
    std::chrono::nanoseconds duration = now - stats->last_time;
    stats->durations[stats->cur_region] += duration;
  }
  stats->last_time = now;
  stats->cur_region = region;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::SearchThread::dump_profiling_stats() {
  profiling_stats_t* stats = get_profiling_stats();
  if (!stats) return;  // compile-time branch

  FILE* f = get_profiling_file();

  fprintf(f, "run_search()\n");
  for (int r = 0; r < int(kNumRegions); ++r) {
    int64_t ns = stats->durations[r].count();
    if (!ns) continue;
    fprintf(f, "%2d %ld\n", r, ns);
  }
  stats->clear();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::SearchThread::profiling_stats_t::clear() {
  for (int r = 0; r < int(kNumRegions); ++r) {
    durations[r] *= 0;
  }
  cur_region = kNumRegions;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts_<GameState, Tensorizor>::NNEvaluationService*
Mcts_<GameState, Tensorizor>::NNEvaluationService::create(const Mcts_* mcts) {
  int64_t timeout_ns = mcts->params().nn_eval_timeout_ns;
  boost::filesystem::path net_filename(mcts->params().nnet_filename);
  size_t cache_size = mcts->params().cache_size;
  int batch_size_limit = mcts->params().batch_size_limit;

  std::chrono::nanoseconds timeout_duration(timeout_ns);
  auto it = instance_map_.find(net_filename);
  if (it == instance_map_.end()) {
    NNEvaluationService* instance = new NNEvaluationService(
        net_filename, batch_size_limit, timeout_duration, cache_size, mcts->profiling_dir());
    instance_map_[net_filename] = instance;
    return instance;
  }
  NNEvaluationService* instance = it->second;
  if (instance->batch_size_limit_ != batch_size_limit) {
    throw util::Exception("Conflicting NNEvaluationService::create() calls: batch_size_limit %d vs %d",
                          instance->batch_size_limit_, batch_size_limit);
  }
  if (instance->timeout_duration_ != timeout_duration) {
    throw util::Exception("Conflicting NNEvaluationService::create() calls: unequal timeout_duration");
  }
  if (instance->cache_.capacity() != cache_size) {
    throw util::Exception("Conflicting NNEvaluationService::create() calls: cache_size %ld vs %ld",
                          instance->cache_.capacity(), cache_size);
  }
  return instance;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::NNEvaluationService::connect() {
  num_connections_++;
  if (thread_) return;
  thread_ = new std::thread([&] { this->loop(); });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::NNEvaluationService::disconnect() {
  if (!thread_) return;

  num_connections_--;
  if (num_connections_ == 0) {
    if (thread_->joinable()) thread_->detach();
    delete thread_;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::NNEvaluationService::NNEvaluationService(
    const boost::filesystem::path& net_filename, int batch_size, std::chrono::nanoseconds timeout_duration,
    size_t cache_size, const boost::filesystem::path& profiling_dir)
: net_(net_filename)
, policy_batch_(batch_size, kNumGlobalActions, util::to_std_array<int>(batch_size, kNumGlobalActions))
, value_batch_(batch_size, kNumPlayers, util::to_std_array<int>(batch_size, kNumPlayers))
, input_batch_(util::to_std_array<int>(batch_size, util::std_array_v<int, typename Tensorizor::Shape>))
, cache_(cache_size)
, timeout_duration_(timeout_duration)
, batch_size_limit_(batch_size)
{
  evaluation_data_batch_ = new evaluation_data_t[batch_size];
  evaluation_pool_.reserve(4096);
  torch_input_gpu_ = input_batch_.asTorch().clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);
  deadline_ = std::chrono::steady_clock::now();

  if (kEnableProfiling) {
    auto profiling_filename = profiling_dir / "eval.txt";
    set_profiling_file(profiling_filename.c_str());
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::NNEvaluationService::~NNEvaluationService() {
  disconnect();
  delete[] evaluation_data_batch_;
  close_profiling_file();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts_<GameState, Tensorizor>::NNEvaluation*
Mcts_<GameState, Tensorizor>::NNEvaluationService::evaluate(
    SearchThread* thread, const Tensorizor& tensorizor, const GameState& state, const ActionMask& valid_action_mask,
    symmetry_index_t sym_index, float inv_temp, bool single_threaded)
{
  thread->record_for_profiling(SearchThread::kCheckingCache);
  cache_key_t key{state, inv_temp, sym_index};

  {
    std::lock_guard<std::mutex> guard(cache_mutex_);
    auto cached = cache_.get(key);
    if (cached.has_value()) {
      return cached.value();
    }
  }

  int my_index;
  {
    thread->record_for_profiling(SearchThread::kAcquiringBatchMutex);
    std::unique_lock<std::mutex> lock(batch_mutex_);
    thread->record_for_profiling(SearchThread::kWaitingUntilBatchReservable);
    cv_evaluate_.wait(lock, [&]{ return batch_reservable(); });
    thread->record_for_profiling(SearchThread::kMisc);

    my_index = batch_reserve_index_;
    assert(my_index < batch_size_limit_);
    batch_reserve_index_++;
    if (my_index == 0) {
      deadline_ = std::chrono::steady_clock::now() + timeout_duration_;
    }
  }
  cv_service_loop_.notify_one();

  /*
   * At this point, the work unit is effectively RESERVED but not COMMITTED.
   *
   * The significance of being reserved is that other search threads will be blocked from reserving if the batch is
   * fully reserved.
   *
   * The significance of not yet being committed is that the service thread won't yet proceed with nnet eval.
   */
  assert(batch_commit_count_ < batch_reserve_index_);
  thread->record_for_profiling(SearchThread::kTensorizing);

  auto& input = input_batch_.template eigenSlab<typename TensorizorTypes::Shape>(my_index);
  tensorizor.tensorize(input, state);
  auto transform = tensorizor.get_symmetry(sym_index);
  transform->transform_input(input);

  evaluation_data_t edata{nullptr, key, valid_action_mask, transform, inv_temp};
  evaluation_data_batch_[my_index] = edata;

  {
    thread->record_for_profiling(SearchThread::kIncrementingCommitCount);
    std::unique_lock<std::mutex> lock(batch_mutex_);
    batch_commit_count_++;
  }
  cv_service_loop_.notify_one();

  NNEvaluation* eval_ptr;
  {
    thread->record_for_profiling(SearchThread::kAcquiringBatchMutex);
    std::unique_lock<std::mutex> lock(batch_mutex_);
    if (single_threaded) batch_evaluate();
    thread->record_for_profiling(SearchThread::kWaitingForReservationProcessing);
    cv_evaluate_.wait(lock, [&]{ return batch_reservations_empty(); });

    eval_ptr = evaluation_data_batch_[my_index].eval_ptr;
    assert(batch_unread_count_ > 0);
    batch_unread_count_--;
  }

  // NOTE: might be able to notify_one(), if we add another notify_one() after the batch_reserve_index_++
  cv_evaluate_.notify_all();
  return eval_ptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::NNEvaluationService::batch_evaluate() {
  assert(batch_reserve_index_ > 0);
  assert(batch_reserve_index_ == batch_commit_count_);

  record_for_profiling(kCopyingCpuToGpu);
  torch_input_gpu_.copy_(input_batch_.asTorch());
  record_for_profiling(kEvaluatingNeuralNet, batch_reserve_index_);
  net_.predict(input_vec_, policy_batch_.asTorch(), value_batch_.asTorch());

  record_for_profiling(kCopyingToPool);
  for (int i = 0; i < batch_reserve_index_; ++i) {
    evaluation_data_t &edata = evaluation_data_batch_[i];
    auto &policy = policy_batch_.eigenSlab(i);
    auto &value = value_batch_.eigenSlab(i);

    edata.transform->transform_policy(policy);
    evaluation_pool_.emplace_back(eigen_util::to_array1d(value), eigen_util::to_array1d(policy),
                                  edata.valid_actions, edata.inv_temp);
    edata.eval_ptr = &evaluation_pool_.back();
  }

  record_for_profiling(kAcquiringCacheMutex);
  {
    std::lock_guard<std::mutex> guard(cache_mutex_);
    record_for_profiling(kFinishingUp);
    for (int i = 0; i < batch_reserve_index_; ++i) {
      const evaluation_data_t &edata = evaluation_data_batch_[i];
      cache_.insert(edata.cache_key, edata.eval_ptr);
    }
  }

  batch_unread_count_ = batch_commit_count_;
  batch_reserve_index_ = 0;
  batch_commit_count_ = 0;
  cv_evaluate_.notify_all();
  record_for_profiling(kNumRegions);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::NNEvaluationService::loop() {
  while (num_connections_) {
    record_for_profiling(kAcquiringBatchMutex);
    std::unique_lock<std::mutex> lock(batch_mutex_);

    record_for_profiling(kWaitingForFirstReservation);
    cv_service_loop_.wait(lock, [&]{ return !batch_reservations_empty(); });
    record_for_profiling(kWaitingForLastReservation);
    cv_service_loop_.wait_until(lock, deadline_, [&]{ return batch_reservations_full(); });
    record_for_profiling(kWaitingForCommits);
    cv_service_loop_.wait(lock, [&]{ return all_batch_reservations_committed(); });

    batch_evaluate();
  }
}

/*
 * The seemingly haphazard combination of macros and runtime-branches for profiling logic is actually carefully
 * concocted! As written, we get the dual benefit of:
 *
 * 1. Zero-branching/pointer-redirection overhead in both profiling and non-profiling mode, thanks to compiler.
 * 2. Compiler checking of profiling methods even when compiled without profiling enabled.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::NNEvaluationService::record_for_profiling(region_t region, int batch_size) {
  profiling_stats_t* stats = get_profiling_stats();
  if (!stats) return;  // compile-time branch

  auto now = std::chrono::steady_clock::now();
  stats->start_times[region] = now;
  if (kEnableVerboseProfiling) {
    int64_t ns = util::ns_since_epoch(now);
    printf("%lu.%09lu eval %d\n", ns / 1000000000, ns % 1000000000, (int) region);
  }

  if (batch_size >= 0) {
    stats->batch_size = batch_size;
  }
  if (region != kNumRegions) return;

  FILE* f = get_profiling_file();
  fprintf(f, "loop size=%d\n", stats->batch_size);
  for (int r = 0; r < int(kNumRegions); ++r) {
    std::chrono::nanoseconds duration = stats->start_times[r + 1] - stats->start_times[r];
    fprintf(f, "%2d %ld\n", r, duration.count());
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::Mcts_(const Params& params)
: params_(params) {
  namespace bf = boost::filesystem;

  nn_eval_service_ = NNEvaluationService::create(this);
  if (num_search_threads() < 1) {
    throw util::Exception("num_search_threads must be positive (%d)", num_search_threads());
  }
  if (params.run_offline && num_search_threads() == 1) {
    throw util::Exception("run_offline does not work with only 1 search thread");
  }
  if (kEnableProfiling && num_search_threads() == 1) {
    throw util::Exception("Profiling mode assumes >1 search thread");
  }
  for (int i = 0; i < num_search_threads(); ++i) {
    search_threads_.push_back(new SearchThread(this, i));
  }

  if (kEnableProfiling) {
    if (profiling_dir().empty()) {
      throw util::Exception("Required: --mcts-profiling-dir. Alternatively, add entry for 'mcts_profiling_dir' in config.txt");
    }
    init_profiling_dir(profiling_dir().string());
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts_<GameState, Tensorizor>::~Mcts_() {
  clear();
  nn_eval_service_->disconnect();
  for (auto* thread : search_threads_) {
    delete thread;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::start() {
  clear();

  if (num_search_threads() == 1) {  // do everything in main-thread
    return;
  }

  nn_eval_service_->connect();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::clear() {
  stop_search_threads();
  if (!root_) return;

  root_->_release();
  delete root_;
  root_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::receive_state_change(
    player_index_t player, const GameState& state, action_index_t action, const GameOutcome& outcome)
{
  stop_search_threads();
  if (!root_) return;

  Node* new_root = root_->_find_child(action);
  if (!new_root) {
    root_->_release();
    delete root_;
    root_ = nullptr;
    return;
  }

  Node* new_root_copy = new Node(*new_root, true);
  root_->_release(new_root);
  delete root_;
  root_ = new_root_copy;
  root_->_adopt_children();

  if (params_.run_offline) {
    start_search_threads(params_.offline_tree_size_limit);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline const typename Mcts_<GameState, Tensorizor>::MctsResults* Mcts_<GameState, Tensorizor>::sim(
    const Tensorizor& tensorizor, const GameState& game_state, const SimParams& params)
{
  stop_search_threads();

  if (!root_ || (!params.disable_noise && params_.dirichlet_mult > 0)) {
    auto outcome = make_non_terminal_outcome<kNumPlayers>();
    symmetry_index_t sym_index = tensorizor.get_random_symmetry_index(game_state);
    root_ = new Node(tensorizor, game_state, outcome, sym_index, params.disable_noise);  // TODO: use memory pool
  }

  start_search_threads(params.tree_size_limit);
  wait_for_search_threads();

  results_.valid_actions = root_->valid_action_mask();
  results_.counts = root_->get_effective_counts();
  results_.policy_prior = root_->_evaluation()->local_policy_prob_distr();
  results_.win_rates = root_->_value_avg();
  results_.value_prior = root_->_evaluation()->value_prob_distr();
  return &results_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::visit(SearchThread* thread, Node* tree, int depth) {
  thread->record_for_profiling(SearchThread::kStartingVisit);
  const auto& outcome = tree->outcome();
  if (is_terminal_outcome(outcome)) {
    thread->record_for_profiling(SearchThread::kBackpropOutcome);
    tree->backprop(outcome);
    if (params_.allow_eliminations) {
      thread->record_for_profiling(SearchThread::kPerformEliminations);
      tree->perform_eliminations(outcome);
    }
    return;
  }

  thread->record_for_profiling(SearchThread::kMisc);
  player_index_t cp = tree->current_player();
  float inv_temp = tree->is_root() ? (1.0 / params_.root_softmax_temperature) : 1.0;
  symmetry_index_t sym_index = tree->sym_index();

  /*
   * Note: race-conditions can cause redundant setting of a given tree's evaluation. This is ok - it just leads to
   * redundant evaluation by the eval thread. It is preferable to inserting a mutex here.
   */
  NNEvaluation* evaluation = tree->_evaluation();
  bool undo_virtual_loss = false;
  if (!evaluation) {
    thread->record_for_profiling(SearchThread::kVirtualBackprop);
    tree->virtual_backprop();
    undo_virtual_loss = true;
    evaluation = nn_eval_service_->evaluate(
        thread, tree->tensorizor(), tree->state(), tree->valid_action_mask(), sym_index, inv_temp,
        num_search_threads() == 1);
    tree->_set_evaluation(evaluation);
  }

  bool leaf = tree->expand_children(thread);

  thread->record_for_profiling(SearchThread::kPUCT);
  auto cPUCT = params_.cPUCT;
  LocalPolicyProbDistr P = evaluation->local_policy_prob_distr();
  const int rows = P.rows();
  assert(rows == int(tree->valid_action_mask().count()));

  using PVec = LocalPolicyProbDistr;

  PVec noise(rows);
  if (tree->is_root() && !tree->disable_noise() && params_.dirichlet_mult) {
    noise = dirichlet_gen_.template generate<LocalPolicyProbDistr>(rng_, params_.dirichlet_alpha, rows);
    P = (1.0 - params_.dirichlet_mult) * P + params_.dirichlet_mult * noise;
  } else {  // TODO - only need to setZero() if generating debug file
    noise.setZero();
  }

  PVec V(rows);
  PVec N(rows);
  PVec E(rows);
  for (int c = 0; c < tree->_num_children(); ++c) {
    Node* child = tree->_get_child(c);
    thread->record_for_profiling(SearchThread::kWaitingForStatsMutex);
    std::lock_guard<std::mutex> guard(child->stats_mutex());
    thread->record_for_profiling(SearchThread::kPUCT);

    V(c) = child->_effective_value_avg(cp);
    N(c) = child->_effective_count();
    E(c) = child->_eliminated();
  }

  constexpr float eps = 1e-6;  // needed when N == 0
  PVec PUCT = V + cPUCT * P * sqrt(N.sum() + eps) / (N + 1);
  PUCT *= 1 - E;

//  // value_avg used for debugging
//  using VVec = ValueArray1D;
//  VVec value_avg;
//  for (int p = 0; p < kNumPlayers; ++p) {
//    value_avg(p) = tree->effective_value_avg(p);
//  }

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);
  Node* best_child = tree->_get_child(argmax_index);

  if (leaf) {
    thread->record_for_profiling(SearchThread::kBackpropEvaluation);
    if (undo_virtual_loss) {
      tree->backprop_with_virtual_undo(evaluation->value_prob_distr());
    } else {
      tree->backprop(evaluation->value_prob_distr());
    }
  } else {
    if (undo_virtual_loss) {
      // unlikely race condition, but possible
      tree->undo_virtual_backprop();
    }
    visit(thread, best_child, depth + 1);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::start_search_threads(int tree_size_limit) {
  assert(!search_active_);
  search_active_ = true;
  num_active_search_threads_ = num_search_threads();

  if (num_search_threads() == 1) {
    // do everything in main thread
    run_search(search_threads_[0], tree_size_limit);
    return;
  }

  for (auto* thread : search_threads_) {
    thread->launch(tree_size_limit);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::wait_for_search_threads() {
  assert(search_active_);
  if (num_search_threads() == 1) return;

  for (auto* thread : search_threads_) {
    thread->join();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::stop_search_threads() {
  search_active_ = false;
  if (num_search_threads() == 1) return;

  std::unique_lock<std::mutex> lock(search_mutex_);
  cv_search_.wait(lock, [&]{ return num_active_search_threads_ == 0; });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts_<GameState, Tensorizor>::run_search(SearchThread* thread, int tree_size_limit) {
  /*
   * Thread-safety analysis:
   *
   * - changes in root_ are always synchronized via stop_search_threads()
   * - _effective_count() and _eliminated() read root_->stats_.{eliminated_, count_}
   *   - eliminated_ starts false and gets flipped to true at most once.
   *   - count_ is monotonoically increasing
   *   - Race-conditions can lead us to read stale values of these. That is ok - that merely causes us to possibly to
   *     more visits than a thread-safe alternative would do.
   */
  while (check_visit_ready(thread, tree_size_limit)) {
    visit(thread, root_, 1);
    thread->dump_profiling_stats();
  }

  std::unique_lock<std::mutex> lock(search_mutex_);
  num_active_search_threads_--;
  cv_search_.notify_one();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts_<GameState, Tensorizor>::init_profiling_dir(const std::string& profiling_dir) {
  static std::string pdir;
  if (!pdir.empty()) {
    if (pdir == profiling_dir) return;
    throw util::Exception("Two different mcts profiling dirs used: %s and %s", pdir.c_str(), profiling_dir.c_str());
  }
  pdir = profiling_dir;

  namespace bf = boost::filesystem;
  bf::path path(profiling_dir);
  if (bf::is_directory(path)) {
    bf::remove_all(path);
  }
  bf::create_directories(path);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline bool Mcts_<GameState, Tensorizor>::check_visit_ready(SearchThread* thread, int tree_size_limit) const {
  thread->record_for_profiling(SearchThread::kCheckVisitReady);
  return search_active_ && root_->_effective_count() <= tree_size_limit && !root_->_eliminated();
}

}  // namespace common
