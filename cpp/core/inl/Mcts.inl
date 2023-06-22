#include <core/Mcts.hpp>

#include <cmath>
#include <thread>
#include <utility>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include <EigenRand/EigenRand>

#include <util/BoostUtil.hpp>
#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>
#include <util/ThreadSafePrinter.hpp>

namespace core {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int Mcts<GameState, Tensorizor>::next_instance_id_ = 0;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int Mcts<GameState, Tensorizor>::NNEvaluationService::next_instance_id_ = 0;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
bool Mcts<GameState, Tensorizor>::NNEvaluationService::session_ended_ = false;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
Mcts<GameState, Tensorizor>::Params::Params(DefaultParamsType type) {
  if (type == kCompetitive) {
    dirichlet_mult = 0;
    dirichlet_alpha_sum = 0;
    forced_playouts = false;
    root_softmax_temperature_str = "1";
  } else if (type == kTraining) {
    root_softmax_temperature_str = "1.4->1.1:2*sqrt(b)";
  } else {
    throw util::Exception("Unknown type: %d", (int)type);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
auto Mcts<GameState, Tensorizor>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  boost::filesystem::path default_profiling_dir_path = util::Repo::root() / "output" / "mcts_profiling";
  std::string default_profiling_dir = util::Config::instance()->get(
      "mcts_profiling_dir", default_profiling_dir_path.string());

  po2::options_description desc("Mcts options");

  return desc
      .template add_option<"model-filename", 'm'>
          (po::value<std::string>(&model_filename),
           "model filename. If not specified, a uniform model is implicitly used")
      .template add_option<"cuda-device">
          (po::value<std::string>(&cuda_device)->default_value(cuda_device), "cuda device")
      .template add_option<"num-search-threads", 'n'>(
          po::value<int>(&num_search_threads)->default_value(num_search_threads),
          "num search threads")
      .template add_option<"batch-size-limit", 'b'>(
          po::value<int>(&batch_size_limit)->default_value(batch_size_limit),
          "batch size limit")
      .template add_bool_switches<"enable-pondering", "disable-pondering">(
          &enable_pondering, "enable pondering (search during opponent's turn)",
          "disable pondering (search during opponent's turn)")
      .template add_option<"pondering-tree-size-limit">(
          po::value<int>(&pondering_tree_size_limit)->default_value(pondering_tree_size_limit),
          "max tree size to grow to when pondering (only respected in --enable-pondering mode)")
      .template add_option<"nn-eval-timeout-ns">(
          po::value<int64_t>(&nn_eval_timeout_ns)->default_value(
          nn_eval_timeout_ns), "nn eval thread timeout in ns")
      .template add_option<"cache-size">(
          po::value<size_t>(&cache_size)->default_value(cache_size),
          "nn eval thread cache size")
      .template add_option<"root-softmax-temp">(
          po::value<std::string>(&root_softmax_temperature_str)->default_value(root_softmax_temperature_str),
          "root softmax temperature")
      .template add_option<"cpuct", 'c'>(po2::float_value("%.2f", &cPUCT), "cPUCT value")
      .template add_option<"dirichlet-mult", 'd'>(po2::float_value("%.2f", &dirichlet_mult), "dirichlet mult")
      .template add_option<"dirichlet-alpha-sum">(po2::float_value("%.2f", &dirichlet_alpha_sum), "dirichlet alpha sum")
      .template add_bool_switches<"disable-eliminations", "enable-eliminations">(
          &disable_eliminations, "disable eliminations", "enable eliminations")
      .template add_bool_switches<"speculative-evals", "no-speculative-evals">(
          &speculative_evals, "enable speculation", "disable speculation")
      .template add_bool_switches<"forced-playouts", "no-forced-playouts">(
          &forced_playouts, "enable forced playouts", "disable forced playouts")
      .template add_bool_switches<"enable-first-play-urgency", "disable-first-play-urgency">(
          &enable_first_play_urgency, "enable first play urgency", "disable first play urgency")
#ifdef PROFILE_MCTS
      .template add_option<"profiling-dir">(po::value<std::string>(&profiling_dir)->default_value(default_profiling_dir),
          "directory in which to dump mcts profiling stats")
#endif  // PROFILE_MCTS
      ;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::NNEvaluationService::instance_map_t
Mcts<GameState, Tensorizor>::NNEvaluationService::instance_map_;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::SearchThread::SearchThread(Mcts* mcts, int thread_id)
: mcts_(mcts)
, params_(mcts->params())
, thread_id_(thread_id) {
  auto profiling_filename = mcts->profiling_dir() / util::create_string("search%d-%d.txt", mcts->instance_id(), thread_id);
  init_profiling(profiling_filename.c_str(), util::create_string("s-%d-%-2d", mcts->instance_id(), thread_id).c_str());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::SearchThread::~SearchThread() {
  kill();
  profiler_t* profiler = get_profiler();
  if (profiler) {
    profiler->dump(get_profiling_file(), 1, get_profiler_name());
  }
  close_profiling_file();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::join() {
  if (thread_ && thread_->joinable()) thread_->join();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::kill() {
  join();
  if (thread_) delete thread_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::launch(const SearchParams* search_params) {
  kill();
  search_params_ = search_params;
  thread_ = new std::thread([&] { mcts_->run_search(this, search_params->tree_size_limit); });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
bool Mcts<GameState, Tensorizor>::SearchThread::needs_more_visits(Node* root, int tree_size_limit) {
  record_for_profiling(kCheckVisitReady);
  const auto& stats = root->stats();
  return mcts_->search_active() && stats.count <= tree_size_limit && !root->eliminated();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::visit(Node* tree, int depth) {
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id());
    printer << __func__ << " " << tree->genealogy_str() << " cp=" << (int)tree->stable_data().current_player;
    printer.endl();
  }

  const auto& stable_data = tree->stable_data();
  const auto& outcome = stable_data.outcome;
  if (is_terminal_outcome(outcome)) {
    backprop_outcome(tree, outcome);
    perform_eliminations(tree, outcome);
    mark_as_fully_analyzed(tree);
    return;
  }

  if (!mcts_->search_active()) return;  // short-circuit

  evaluate_and_expand_result_t data = evaluate_and_expand(tree, false);
  NNEvaluation* evaluation = data.evaluation.get();
  assert(evaluation);

  if (data.backpropagated_virtual_loss) {
    record_for_profiling(kBackpropEvaluation);

    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread_id());
      printer << "backprop_with_virtual_undo " << tree->genealogy_str();
      printer << " " << evaluation->value_prob_distr().transpose();
      printer.endl();
    }

    tree->backprop_with_virtual_undo(evaluation->value_prob_distr());
  } else {
    child_index_t best_child_index = get_best_child_index(tree, evaluation);
    Node* child = tree->init_child(best_child_index);
    visit(child, depth + 1);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::backprop_outcome(Node* tree, const ValueArray& outcome) {
  record_for_profiling(kBackpropOutcome);
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << tree->genealogy_str() << " " << outcome.transpose();
    printer.endl();
  }

  tree->backprop(outcome);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::perform_eliminations(Node* tree, const ValueArray& outcome) {
  if (params_.disable_eliminations) return;
  player_bitset_t forcibly_winning;
  player_bitset_t forcibly_losing;
  for (int p = 0; p < kNumPlayers; ++p) {
    forcibly_winning.set(p, outcome(p) == 1);
    forcibly_losing.set(p, outcome(p) == 0);
  }
  int cp = tree->stable_data().current_player;
  bool winning = outcome(cp) == 1;
  bool losing = outcome(cp) == 0;
  if (!winning && !losing) return;  // drawn position, no elimination possible

  ValueArray accumulated_value;
  accumulated_value.setZero();
  int accumulated_count = 0;

  record_for_profiling(kPerformEliminations);
  tree->eliminate(thread_id_, forcibly_winning, forcibly_losing, accumulated_value, accumulated_count);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::mark_as_fully_analyzed(Node* tree) {
  record_for_profiling(kMarkFullyAnalyzed);
  tree->mark_as_fully_analyzed();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::SearchThread::evaluate_and_expand_result_t
Mcts<GameState, Tensorizor>::SearchThread::evaluate_and_expand(Node* tree, bool speculative) {
  record_for_profiling(kEvaluateAndExpand);

  std::unique_lock<std::mutex> lock(tree->evaluation_data_mutex());
  typename Node::evaluation_data_t& evaluation_data = tree->evaluation_data();
  evaluate_and_expand_result_t data{evaluation_data.ptr.load(), false};
  auto state = evaluation_data.state;

  switch (state) {
    case Node::kUnset:
    {
      evaluate_and_expand_unset(tree, &lock, &data, speculative);
      tree->cv_evaluate_and_expand().notify_all();
      break;
    }
    case Node::kPending:
    {
      assert(params_.speculative_evals);
      evaluate_and_expand_pending(tree, &lock);
      assert(!lock.owns_lock());
      if (!speculative) {
        lock.lock();
        if (evaluation_data.state != Node::kSet) {
          tree->cv_evaluate_and_expand().wait(lock);
          assert(evaluation_data.state == Node::kSet);
        }
        data.evaluation = evaluation_data.ptr.load();
        assert(data.evaluation.get());
      }
      break;
    }
    default: break;
  }
  return data;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::SearchThread::evaluate_and_expand_unset(
    Node* tree, std::unique_lock<std::mutex>* lock, evaluate_and_expand_result_t* data, bool speculative)
{
  record_for_profiling(kEvaluateAndExpandUnset);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << tree->genealogy_str();
    printer.endl();
  }

  assert(!tree->has_children());
  data->backpropagated_virtual_loss = true;
  assert(data->evaluation.get() == nullptr);

  auto& evaluation_data = tree->evaluation_data();

  if (params_.speculative_evals) {
    evaluation_data.state = Node::kPending;
    lock->unlock();
  }

  if (!speculative) {
    record_for_profiling(kVirtualBackprop);
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread_id_);
      printer << "virtual_backprop " << tree->genealogy_str();
      printer.endl();
    }

    tree->virtual_backprop();
  }

  const auto& stable_data = tree->stable_data();
  bool used_cache = false;
  if (!mcts_->nn_eval_service()) {
    // no-model mode
    ValueTensor uniform_value;
    PolicyTensor uniform_policy;
    uniform_value.setConstant(1.0 / kNumPlayers);
    uniform_policy.setConstant(0);
    data->evaluation = std::make_shared<NNEvaluation>(uniform_value, uniform_policy, stable_data.valid_action_mask);
  } else {
    symmetry_index_t sym_index = stable_data.sym_index;
    typename NNEvaluationService::Request request{this, tree, sym_index};
    auto response = mcts_->nn_eval_service()->evaluate(request);
    used_cache = response.used_cache;
    data->evaluation = response.ptr;
  }

  if (params_.speculative_evals) {
    if (speculative && used_cache) {
      // without this, when we hit cache, we fail to saturate nn service batch
      lock->lock();
      evaluate_and_expand_pending(tree, lock);
    }

    lock->lock();
  }

  LocalPolicyArray P = eigen_util::softmax(data->evaluation->local_policy_logit_distr());
  if (tree->is_root()) {
    if (!search_params_->disable_exploration) {
      if (params_.dirichlet_mult) {
        mcts_->add_dirichlet_noise(P);
      }
      P = P.pow(1.0 / mcts_->root_softmax_temperature());
      P /= P.sum();
    }
  }
  evaluation_data.local_policy_prob_distr = P;
  evaluation_data.ptr.store(data->evaluation);
  evaluation_data.state = Node::kSet;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::SearchThread::evaluate_and_expand_pending(
    Node* tree, std::unique_lock<std::mutex>* lock)
{
  // Another search thread is working on this. Might as well speculatively eval another position while we wait
  record_for_profiling(kEvaluateAndExpandPending);

  assert(tree->has_children());
  Node* child;
  auto& evaluation_data = tree->evaluation_data();
  if (evaluation_data.fully_analyzed_actions.all()) {
    child = tree->get_child(0);
    assert(child);
    lock->unlock();
  } else {
    action_index_t action = bitset_util::choose_random_off_index(evaluation_data.fully_analyzed_actions);
    lock->unlock();
    child = tree->lookup_child_by_action(action);
    assert(child);
  }

  const auto& stable_data = child->stable_data();
  const auto& outcome = stable_data.outcome;
  if (is_terminal_outcome(outcome)) {
    perform_eliminations(child, outcome);  // why not?
    mark_as_fully_analyzed(child);
  } else {
    evaluate_and_expand(child, true);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
Mcts<GameState, Tensorizor>::child_index_t
Mcts<GameState, Tensorizor>::SearchThread::get_best_child_index(Node* tree, NNEvaluation* evaluation) {
  record_for_profiling(kPUCT);

  PUCTStats stats(params_, *search_params_, tree);

  using PVec = LocalPolicyArray;

  const PVec& P = stats.P;
  const PVec& N = stats.N;
  const PVec& VN = stats.VN;
  const PVec& E = stats.E;
  PVec& PUCT = stats.PUCT;

  bool add_noise = !search_params_->disable_exploration && params_.dirichlet_mult > 0;
  if (params_.forced_playouts && add_noise) {
    PVec n_forced = (P * params_.k_forced * N.sum()).sqrt();
    auto F1 = (N < n_forced).template cast<dtype>();
    auto F2 = (N > 0).template cast<dtype>();
    auto F = F1 * F2;
    PUCT = PUCT * (1 - F) + F * 1e+6;
  }

  PUCT -= E * (100 + PUCT.maxCoeff() - PUCT.minCoeff());  // zero-out where E==1

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);

  mcts_->record_puct_calc(VN.sum() > 0);

  if (mcts::kEnableThreadingDebug) {
    std::string genealogy = tree->genealogy_str();

    util::ThreadSafePrinter printer(thread_id());

    printer << "*************";
    printer.endl();
    printer << __func__ << "() " << genealogy;
    printer.endl();
    printer << "valid:";
    for (int v : bitset_util::on_indices(tree->stable_data().valid_action_mask)) {
      printer << " " << v;
    }
    printer.endl();
    printer << "value_avg: " << tree->stats().value_avg.transpose();
    printer.endl();
    printer << "P: " << P.transpose();
    printer.endl();
    printer << "N: " << N.transpose();
    printer.endl();
    printer << "V: " << stats.V.transpose();
    printer.endl();
    printer << "VN: " << stats.VN.transpose();
    printer.endl();
    printer << "E: " << E.transpose();
    printer.endl();
    printer << "PUCT: " << PUCT.transpose();
    printer.endl();
    printer << "argmax: " << argmax_index;
    printer.endl();
    printer << "*************";
    printer.endl();
  }
  return argmax_index;
}

/*
 * The seemingly haphazard combination of macros and runtime-branches for profiling logic is actually carefully
 * concocted! As written, we get the dual benefit of:
 *
 * 1. Zero-branching/pointer-redirection overhead in both profiling and non-profiling mode, thanks to compiler.
 * 2. Compiler checking of profiling methods even when compiled without profiling enabled.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::record_for_profiling(region_t region) {
  profiler_t* profiler = get_profiler();
  if (!profiler) return;  // compile-time branch
  profiler->record(region, get_profiler_name());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::dump_profiling_stats() {
  profiler_t* profiler = get_profiler();
  if (!profiler) return;  // compile-time branch
  profiler->dump(get_profiling_file(), 64, get_profiler_name());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::PUCTStats::PUCTStats(
    const Params& params, const SearchParams& search_params, const Node* tree)
: cp(tree->stable_data().current_player)
, P(tree->evaluation_data().local_policy_prob_distr)
, V(P.rows())
, N(P.rows())
, VN(P.rows())
, E(P.rows())
, PUCT(P.rows())
{
  V.setZero();
  N.setZero();
  VN.setZero();
  E.setZero();

  std::bitset<kMaxNumLocalActions> fpu_bits;

  for (child_index_t c = 0; c < tree->stable_data().num_valid_actions(); ++c) {
    /*
     * NOTE: we do NOT grab the child stats_mutex here! This means that child_stats can contain
     * arbitrarily-partially-written data.
     */
    Node* child = tree->get_child(c);
    if (!child) {
      fpu_bits[c] = true;
      continue;
    }
    auto child_stats = child->stats();  // struct copy to simplify reasoning about race conditions

    V(c) = child_stats.value_avg(cp);
    N(c) = child_stats.count;
    VN(c) = child_stats.virtual_count;
    E(c) = child->eliminated(child_stats);

    fpu_bits[c] = (N(c) == 0);
  }

  if (params.enable_first_play_urgency && fpu_bits.any()) {
    /*
     * Again, we do NOT grab the stats_mutex here!
     */
    const auto& stats = tree->stats();  // no struct copy, not needed here
    dtype PV = stats.value_avg(cp);

    bool disableFPU = tree->is_root() && params.dirichlet_mult > 0 && !search_params.disable_exploration;
    dtype cFPU = disableFPU ? 0.0 : params.cFPU;
    dtype v = PV - cFPU * sqrt((P * (N > 0).template cast<dtype>()).sum());
    for (int c : bitset_util::on_indices(fpu_bits)) {
      V(c) = v;
    }
  }

  /*
   * AlphaZero/KataGo defines V to be over a [-1, +1] range, but we use a [0, +1] range.
   *
   * We multiply V by 2 to account for this difference.
   *
   * This could have been accomplished also by multiplying cPUCT by 0.5, but this way maintains better
   * consistency with the AlphaZero/KataGo approach.
   */
  PUCT = 2 * V + params.cPUCT * P * sqrt(N.sum() + eps) / (N + 1);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::NNEvaluationService*
Mcts<GameState, Tensorizor>::NNEvaluationService::create(const Mcts* mcts) {
  int64_t timeout_ns = mcts->params().nn_eval_timeout_ns;
  boost::filesystem::path model_filename(mcts->params().model_filename);
  size_t cache_size = mcts->params().cache_size;
  int batch_size_limit = mcts->params().batch_size_limit;

  std::chrono::nanoseconds timeout_duration(timeout_ns);
  auto it = instance_map_.find(model_filename);
  if (it == instance_map_.end()) {
    NNEvaluationService* instance = new NNEvaluationService(
        model_filename, mcts->params().cuda_device, batch_size_limit, timeout_duration, cache_size,
        mcts->profiling_dir());
    instance_map_[model_filename] = instance;
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
void Mcts<GameState, Tensorizor>::NNEvaluationService::connect() {
  std::lock_guard<std::mutex> guard(connection_mutex_);
  num_connections_++;
  if (thread_) return;
  thread_ = new std::thread([&] { this->loop(); });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::disconnect() {
  std::lock_guard<std::mutex> guard(connection_mutex_);
  if (thread_) {
    num_connections_--;
    if (num_connections_ > 0) return;
    if (thread_->joinable()) thread_->detach();
    delete thread_;
    thread_ = nullptr;
  }
  close_profiling_file();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationService::NNEvaluationService(
    const boost::filesystem::path& model_filename, const std::string& cuda_device, int batch_size,
    std::chrono::nanoseconds timeout_duration, size_t cache_size, const boost::filesystem::path& profiling_dir)
: instance_id_(next_instance_id_++)
, net_(model_filename, cuda_device)
, batch_data_(batch_size)
, full_input_(util::to_std_array<int64_t>(batch_size, eigen_util::to_int64_std_array_v<InputShape>))
, cache_(cache_size)
, timeout_duration_(timeout_duration)
, batch_size_limit_(batch_size)
{
  auto input_shape = util::to_std_array<int64_t>(batch_size, eigen_util::to_int64_std_array_v<InputShape>);
  auto policy_shape = util::to_std_array<int64_t>(batch_size, eigen_util::to_int64_std_array_v<PolicyShape>);
  auto value_shape = util::to_std_array<int64_t>(batch_size, eigen_util::to_int64_std_array_v<ValueShape>);

  torch_input_gpu_ = torch::empty(input_shape, torch_util::to_dtype_v<dtype>).to(at::Device(cuda_device));
  torch_policy_ = torch::empty(policy_shape, torch_util::to_dtype_v<PolicyScalar>);
  torch_value_ = torch::empty(value_shape, torch_util::to_dtype_v<ValueScalar>);

  input_vec_.push_back(torch_input_gpu_);
  deadline_ = std::chrono::steady_clock::now();

  std::string name = util::create_string("eval-%d", instance_id_);
  auto profiling_filename = profiling_dir / util::create_string("%s.txt", name.c_str());
  init_profiling(profiling_filename.c_str(), name.c_str());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::NNEvaluationService::tensor_group_t::load_output_from(
    int row, torch::Tensor& torch_policy, torch::Tensor& torch_value)
{
  constexpr size_t policy_size = PolicyShape::total_size;
  constexpr size_t value_size = ValueShape::total_size;

  memcpy(policy.data(), torch_policy.data_ptr<PolicyScalar>() + row * policy_size, policy_size * sizeof(PolicyScalar));
  memcpy(value.data(), torch_value.data_ptr<ValueScalar>() + row * value_size, value_size * sizeof(ValueScalar));
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationService::batch_data_t::batch_data_t(int batch_size) {
  tensor_groups_ = new tensor_group_t[batch_size];
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationService::batch_data_t::~batch_data_t() {
  delete[] tensor_groups_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::NNEvaluationService::batch_data_t::copy_input_to(
    int num_rows, DynamicInputFloatTensor& full_input)
{
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

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationService::~NNEvaluationService() {
  disconnect();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::NNEvaluationService::Response
Mcts<GameState, Tensorizor>::NNEvaluationService::evaluate(const Request& request) {
  SearchThread* thread = request.thread;
  const Node* tree = request.tree;

  if (mcts::kEnableThreadingDebug) {
    std::string genealogy = tree->genealogy_str();
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("evaluate() %s\n", genealogy.c_str());
  }

  const auto& stable_data = tree->stable_data();
  cache_key_t cache_key{stable_data.state, request.sym_index};
  Response response = check_cache(thread, cache_key);
  if (response.used_cache) return response;

  std::unique_lock<std::mutex> metadata_lock(batch_metadata_.mutex);
  wait_until_batch_reservable(thread, metadata_lock);
  int my_index = allocate_reserve_index(thread, metadata_lock);
  metadata_lock.unlock();

  tensorize_and_transform_input(request, cache_key, my_index);

  metadata_lock.lock();
  increment_commit_count(thread);
  NNEvaluation_sptr eval_ptr = get_eval(thread, my_index, metadata_lock);
  wait_until_all_read(thread, metadata_lock);
  metadata_lock.unlock();

  cv_evaluate_.notify_all();

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  evaluated!\n");
  }

  return Response{eval_ptr, false};
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::get_cache_stats(
    int& hits, int& misses, int& size, float& hash_balance_factor) const
{
  hits = cache_hits_;
  misses = cache_misses_;
  size = cache_.size();
  hash_balance_factor = cache_.get_hash_balance_factor();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::record_puct_calc(bool virtual_loss_influenced) {
  this->total_puct_calcs_++;
  if (virtual_loss_influenced) {
    this->virtual_loss_influenced_puct_calcs_++;
  }
}


template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::end_session() {
  if (session_ended_) return;

  int64_t evaluated_positions = 0;
  int64_t batches_evaluated = 0;
  for (auto it : instance_map_) {
    NNEvaluationService* service = it.second;
    evaluated_positions += service->evaluated_positions_;
    batches_evaluated += service->batches_evaluated_;
  }

  float avg_batch_size = batches_evaluated > 0 ? evaluated_positions * 1.0 / batches_evaluated : 0.0f;

  util::ParamDumper::add("MCTS evaluated positions", "%ld", evaluated_positions);
  util::ParamDumper::add("MCTS batches evaluated", "%ld", batches_evaluated);
  util::ParamDumper::add("MCTS avg batch size", "%.2f", avg_batch_size);
  session_ended_ = true;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
float Mcts<GameState, Tensorizor>::NNEvaluationService::pct_virtual_loss_influenced_puct_calcs() {
  int64_t num = 0;
  int64_t den = 0;

  for (auto it : instance_map_) {
    NNEvaluationService* service = it.second;
    num += service->virtual_loss_influenced_puct_calcs_;
    den += service->total_puct_calcs_;
  }

  return 100.0 * num / den;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::batch_evaluate() {
  std::unique_lock batch_metadata_lock(batch_metadata_.mutex);
  std::unique_lock batch_data_lock(batch_data_.mutex);

  assert(batch_metadata_.reserve_index > 0);
  assert(batch_metadata_.reserve_index == batch_metadata_.commit_count);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- NNEvaluationService::%s(%s) ---------------------->\n",
                   __func__, batch_metadata_.repr().c_str());
  }

  record_for_profiling(kCopyingCpuToGpu);
  int num_rows = batch_metadata_.reserve_index;
  batch_data_.copy_input_to(num_rows, full_input_);
  auto input_shape = util::to_std_array<int64_t>(num_rows, eigen_util::to_int64_std_array_v<InputShape>);
  torch::Tensor full_input_torch = torch::from_blob(full_input_.data(), input_shape);
  torch_input_gpu_.resize_(input_shape);
  torch_input_gpu_.copy_(full_input_torch);

  record_for_profiling(kEvaluatingNeuralNet);
  torch_policy_.resize_(util::to_std_array<int64_t>(num_rows, eigen_util::to_int64_std_array_v<PolicyShape>));
  torch_value_.resize_(util::to_std_array<int64_t>(num_rows, eigen_util::to_int64_std_array_v<ValueShape>));
  net_.predict(input_vec_, torch_policy_, torch_value_);

  record_for_profiling(kCopyingToPool);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    tensor_group_t& group = batch_data_.tensor_groups_[i];
    group.load_output_from(i, torch_policy_, torch_value_);
    eval_ptr_data_t& edata = group.eval_ptr_data;

    eigen_util::right_rotate(eigen_util::reinterpret_as_array(group.value), group.current_player);
    edata.transform->transform_policy(group.policy);
    edata.eval_ptr.store(std::make_shared<NNEvaluation>(group.value, group.policy, edata.valid_actions));
  }

  record_for_profiling(kAcquiringCacheMutex);
  std::unique_lock<std::mutex> lock(cache_mutex_);
  record_for_profiling(kFinishingUp);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    const eval_ptr_data_t &edata = batch_data_.tensor_groups_[i].eval_ptr_data;
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

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::loop() {
  while (active()) {
    wait_until_batch_ready();
    wait_for_first_reservation();
    wait_for_last_reservation();
    wait_for_commits();
    batch_evaluate();
    dump_profiling_stats();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
Mcts<GameState, Tensorizor>::NNEvaluationService::Response
Mcts<GameState, Tensorizor>::NNEvaluationService::check_cache(SearchThread* thread, const cache_key_t& cache_key) {
  thread->record_for_profiling(SearchThread::kCheckingCache);

  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  waiting for cache lock...\n");
  }

  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  auto cached = cache_.get(cache_key);
  if (cached.has_value()) {
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread->thread_id());
      printer.printf("  hit cache\n");
    }
    cache_hits_++;
    return Response{cached.value(), true};
  }
  cache_misses_++;
  return Response{nullptr, false};
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::wait_until_batch_reservable(
    SearchThread* thread, std::unique_lock<std::mutex>& metadata_lock)
{
  thread->record_for_profiling(SearchThread::kWaitingUntilBatchReservable);

  const char* func = __func__;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&]{
    if (batch_metadata_.unread_count == 0 && batch_metadata_.reserve_index < batch_size_limit_ && batch_metadata_.accepting_reservations) return true;
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread->thread_id());
      printer.printf("  %s(%s) still waiting...\n", func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int Mcts<GameState, Tensorizor>::NNEvaluationService::allocate_reserve_index(
    SearchThread* thread, std::unique_lock<std::mutex>& metadata_lock)
{
  thread->record_for_profiling(SearchThread::kMisc);

  int my_index = batch_metadata_.reserve_index;
  assert(my_index < batch_size_limit_);
  batch_metadata_.reserve_index++;
  if (my_index == 0) {
    deadline_ = std::chrono::steady_clock::now() + timeout_duration_;
  }
  assert(batch_metadata_.commit_count < batch_metadata_.reserve_index);

  const char* func = __func__;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s) allocation complete\n", func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.notify_one();

  /*
   * At this point, the work unit is effectively RESERVED but not COMMITTED.
   *
   * The significance of being reserved is that other search threads will be blocked from reserving if the batch is
   * fully reserved.
   *
   * The significance of not yet being committed is that the service thread won't yet proceed with model eval.
   */
  return my_index;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::tensorize_and_transform_input(
    const Request& request, const cache_key_t& cache_key, int reserve_index)
{
  SearchThread* thread = request.thread;
  Node* tree = request.tree;

  const auto& stable_data = tree->stable_data();
  const Tensorizor& tensorizor = stable_data.tensorizor;
  const GameState& state = stable_data.state;
  const ActionMask& valid_action_mask = stable_data.valid_action_mask;
  seat_index_t current_player = stable_data.current_player;
  symmetry_index_t sym_index = cache_key.sym_index;

  thread->record_for_profiling(SearchThread::kTensorizing);
  std::unique_lock<std::mutex> lock(batch_data_.mutex);

  tensor_group_t& group = batch_data_.tensor_groups_[reserve_index];
  tensorizor.tensorize(group.input, state);
  auto transform = tensorizor.get_symmetry(sym_index);
  transform->transform_input(group.input);

  group.current_player = current_player;
  group.eval_ptr_data.eval_ptr.store(nullptr);
  group.eval_ptr_data.cache_key = cache_key;
  group.eval_ptr_data.valid_actions = valid_action_mask;
  group.eval_ptr_data.transform = transform;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::increment_commit_count(SearchThread* thread)
{
  thread->record_for_profiling(SearchThread::kIncrementingCommitCount);

  batch_metadata_.commit_count++;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", __func__, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.notify_one();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::NNEvaluation_sptr
Mcts<GameState, Tensorizor>::NNEvaluationService::get_eval(
    SearchThread* thread, int reserve_index, std::unique_lock<std::mutex>& metadata_lock)
{
  const char* func = __func__;
  thread->record_for_profiling(SearchThread::kWaitingForReservationProcessing);
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&]{
    if (batch_metadata_.reserve_index == 0) return true;
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread->thread_id());
      printer.printf("  %s(%s) still waiting...\n", func, batch_metadata_.repr().c_str());
    }
    return false;
  });

  return batch_data_.tensor_groups_[reserve_index].eval_ptr_data.eval_ptr.load();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::wait_until_all_read(
    SearchThread* thread, std::unique_lock<std::mutex>& metadata_lock)
{
  assert(batch_metadata_.unread_count > 0);
  batch_metadata_.unread_count--;

  const char* func = __func__;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&]{
    if (batch_metadata_.unread_count == 0) return true;
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread->thread_id());
      printer.printf("  %s(%s) still waiting...\n", func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::wait_until_batch_ready()
{
  record_for_profiling(kWaitingUntilBatchReady);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&]{
    if (batch_metadata_.unread_count == 0) return true;
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::wait_for_first_reservation()
{
  record_for_profiling(kWaitingForFirstReservation);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&]{
    if (batch_metadata_.reserve_index > 0) return true;
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::wait_for_last_reservation()
{
  record_for_profiling(kWaitingForLastReservation);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait_until(lock, deadline_, [&]{
    if (batch_metadata_.reserve_index == batch_size_limit_) return true;
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
    }
    return false;
  });
  batch_metadata_.accepting_reservations = false;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::wait_for_commits()
{
  record_for_profiling(kWaitingForCommits);
  std::unique_lock<std::mutex> lock(batch_metadata_.mutex);
  const char* cls = "NNEvaluationService";
  const char* func = __func__;
  if (mcts::kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&]{
    if (batch_metadata_.reserve_index == batch_metadata_.commit_count) return true;
    if (mcts::kEnableThreadingDebug) {
      util::ThreadSafePrinter printer;
      printer.printf("<---------------------- %s %s(%s) still waiting ---------------------->\n",
                     cls, func, batch_metadata_.repr().c_str());
    }
    return false;
  });
}

/*
 * The seemingly haphazard combination of macros and runtime-branches for profiling logic is actually carefully
 * concocted! As written, we get the dual benefit of:
 *
 * 1. Zero-branching/pointer-redirection overhead in both profiling and non-profiling mode, thanks to compiler.
 * 2. Compiler checking of profiling methods even when compiled without profiling enabled.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::record_for_profiling(region_t region) {
  profiler_t* profiler = get_profiler();
  if (!profiler) return;  // compile-time branch
  profiler->record(region, get_profiler_name());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::dump_profiling_stats() {
  profiler_t* profiler = get_profiler();
  if (!profiler) return;  // compile-time branch
  profiler->dump(get_profiling_file(), 64, get_profiler_name());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::NodeReleaseService Mcts<GameState, Tensorizor>::NodeReleaseService::instance_;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
Mcts<GameState, Tensorizor>::NodeReleaseService::NodeReleaseService()
: thread_([&] { loop();})
{
  struct sched_param param;
  param.sched_priority = 0;
  pthread_setschedparam(thread_.native_handle(), SCHED_IDLE, &param);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
Mcts<GameState, Tensorizor>::NodeReleaseService::~NodeReleaseService() {
  destructing_ = true;
  cv_.notify_one();
  if (thread_.joinable()) thread_.join();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NodeReleaseService::loop() {
  while (!destructing_) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_queue_t& queue = work_queue_[queue_index_];
    cv_.wait(lock, [&]{ return !queue.empty() || destructing_;});
    if (destructing_) return;
    queue_index_ = 1 - queue_index_;
    lock.unlock();
    for (auto& unit : queue) {
      unit.node->release(unit.arg);
    }
    queue.clear();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NodeReleaseService::release_helper(Node* node, Node* arg) {
  std::unique_lock<std::mutex> lock(mutex_);
  work_queue_t& queue = work_queue_[queue_index_];
  queue.emplace_back(node, arg);
  max_queue_size_ = std::max(max_queue_size_, int(queue.size()));
  lock.unlock();
  cv_.notify_one();
  release_count_++;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Mcts(const Params& params)
: params_(params)
, pondering_search_params_(SearchParams::make_pondering_params(params.pondering_tree_size_limit))
, instance_id_(next_instance_id_++)
, root_softmax_temperature_(math::ExponentialDecay::parse(
    params.root_softmax_temperature_str, GameStateTypes::get_var_bindings()))
{
  namespace bf = boost::filesystem;

  if (mcts::kEnableProfiling) {
    if (profiling_dir().empty()) {
      throw util::Exception("Required: --mcts-profiling-dir. Alternatively, add entry for 'mcts_profiling_dir' in config.txt");
    }
    init_profiling_dir(profiling_dir().string());
  }

  if (!params.model_filename.empty()) {
    nn_eval_service_ = NNEvaluationService::create(this);
  }
  if (num_search_threads() < 1) {
    throw util::Exception("num_search_threads must be positive (%d)", num_search_threads());
  }
  if (params.enable_pondering && num_search_threads() == 1) {
    throw util::Exception("pondering mode does not work with only 1 search thread");
  }
  for (int i = 0; i < num_search_threads(); ++i) {
    search_threads_.push_back(new SearchThread(this, i));
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::~Mcts() {
  clear();
  if (nn_eval_service_) {
    nn_eval_service_->disconnect();
  }
  for (auto* thread : search_threads_) {
    delete thread;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::start() {
  clear();
  root_softmax_temperature_.reset();

  if (!connected_) {
    if (nn_eval_service_) {
      nn_eval_service_->connect();
    }
    connected_ = true;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::clear() {
  stop_search_threads();
  if (!root_) return;

  assert(root_->parent()==nullptr);
  NodeReleaseService::release(root_);
  root_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::receive_state_change(
    seat_index_t seat, const GameState& state, action_index_t action)
{
  root_softmax_temperature_.step();
  stop_search_threads();
  if (!root_) return;

  assert(root_->parent()==nullptr);

  Node* new_root = root_->lookup_child_by_action(action);
  if (!new_root) {
    NodeReleaseService::release(root_);
    root_ = nullptr;
    return;
  }

  Node* new_root_copy = new Node(*new_root, true);
  NodeReleaseService::release(root_, new_root);
  root_ = new_root_copy;
  root_->adopt_children();

  if (params_.enable_pondering) {
    start_search_threads(&pondering_search_params_);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline const typename Mcts<GameState, Tensorizor>::MctsResults* Mcts<GameState, Tensorizor>::search(
    const Tensorizor& tensorizor, const GameState& game_state, const SearchParams& params)
{
  stop_search_threads();

  bool add_noise = !params.disable_exploration && params_.dirichlet_mult > 0;
  if (!root_ || add_noise) {
    if (root_) {
      NodeReleaseService::release(root_);
    }
    auto outcome = make_non_terminal_outcome<kNumPlayers>();
    root_ = new Node(tensorizor, game_state, outcome);  // TODO: use memory pool
  }

  start_search_threads(&params);
  wait_for_search_threads();

  const auto& evaluation_data = root_->evaluation_data();
  const auto& stable_data = root_->stable_data();

  NNEvaluation_sptr evaluation = evaluation_data.ptr.load();
  results_.valid_actions = stable_data.valid_action_mask;
  results_.counts = root_->get_counts();
  if (params_.forced_playouts && add_noise) {
    prune_counts(params);
  }
  results_.policy_prior = evaluation_data.local_policy_prob_distr;
  results_.win_rates = root_->stats().value_avg;
  results_.value_prior = evaluation->value_prob_distr();
  return &results_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::add_dirichlet_noise(LocalPolicyArray& P) {
  int rows = P.rows();
  double alpha = params_.dirichlet_alpha_sum / rows;
  LocalPolicyArray noise = dirichlet_gen_.template generate<LocalPolicyArray>(rng_, alpha, rows);
  P = (1.0 - params_.dirichlet_mult) * P + params_.dirichlet_mult * noise;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::start_search_threads(const SearchParams* search_params) {
  assert(!search_active_);
  search_active_ = true;
  num_active_search_threads_ = num_search_threads();

  for (auto* thread : search_threads_) {
    thread->launch(search_params);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::wait_for_search_threads() {
  assert(search_active_);

  for (auto* thread : search_threads_) {
    thread->join();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::stop_search_threads() {
  search_active_ = false;

  std::unique_lock<std::mutex> lock(search_mutex_);
  cv_search_.wait(lock, [&]{ return num_active_search_threads_ == 0; });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::run_search(SearchThread* thread, int tree_size_limit) {
  thread->visit(root_, 1);
  thread->dump_profiling_stats();

  if (!thread->is_pondering() && root_->stable_data().num_valid_actions() > 1) {
    while (thread->needs_more_visits(root_, tree_size_limit)) {
      thread->visit(root_, 1);
      thread->dump_profiling_stats();
    }
  }

  std::unique_lock<std::mutex> lock(search_mutex_);
  num_active_search_threads_--;
  cv_search_.notify_one();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::get_cache_stats(
    int& hits, int& misses, int& size, float& hash_balance_factor) const
{
  nn_eval_service_->get_cache_stats(hits, misses, size, hash_balance_factor);
}

/*
 * The KataGo paper is a little vague in its description of the target pruning step, and examining the KataGo
 * source code was not very enlightening. The following is my best guess at what the target pruning step does.
 */
template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::prune_counts(const SearchParams& search_params) {
  if (params_.model_filename.empty()) return;

  PUCTStats stats(params_, search_params, root_);

  auto orig_counts = results_.counts;
  const auto& P = stats.P;
  const auto& N = stats.N;
  const auto& V = stats.V;
  const auto& PUCT = stats.PUCT;

  auto N_sum = N.sum();
  auto n_forced = (P * params_.k_forced * N_sum).sqrt();

  auto PUCT_max = PUCT.maxCoeff();

  auto N_max = N.maxCoeff();
  auto sqrt_N = sqrt(N_sum + PUCTStats::eps);

  auto N_floor = params_.cPUCT * P * sqrt_N / (PUCT_max - 2 * V) - 1;
  for (child_index_t c = 0; c < root_->stable_data().num_valid_actions(); ++c) {
    if (N(c) == N_max) continue;
    if (!isfinite(N_floor(c))) continue;
    auto n = std::max(N_floor(c), N(c) - n_forced(c));
    if (n <= 1.0) {
      n = 0;
    }

    Node* child = root_->get_child(c);
    if (child) {
      results_.counts(child->action()) = n;
    }
  }

  const auto& counts_array = eigen_util::reinterpret_as_array(results_.counts);
  if (counts_array.sum() <= 0) {
    // can happen in certain edge cases
    results_.counts = orig_counts;
    return;
  }

  if (!counts_array.isFinite().all()) {
    std::cout << "P: " << P.transpose() << std::endl;
    std::cout << "N: " << N.transpose() << std::endl;
    std::cout << "V: " << V.transpose() << std::endl;
    std::cout << "PUCT: " << PUCT.transpose() << std::endl;
    std::cout << "n_forced: " << n_forced.transpose() << std::endl;
    std::cout << "orig_counts: " << eigen_util::reinterpret_as_array(orig_counts).transpose() << std::endl;
    std::cout << "results_.counts: " << counts_array.transpose() << std::endl;
    throw util::Exception("prune_counts: counts problem");
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::init_profiling_dir(const std::string& profiling_dir) {
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

}  // namespace core