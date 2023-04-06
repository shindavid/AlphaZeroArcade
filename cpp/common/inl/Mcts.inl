#include <common/Mcts.hpp>

#include <cmath>
#include <thread>
#include <utility>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include <EigenRand/EigenRand>

#include <util/BoostUtil.hpp>
#include <util/Config.hpp>
#include <util/EigenTorch.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>
#include <util/StringUtil.hpp>
#include <util/ThreadSafePrinter.hpp>

namespace common {

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int Mcts<GameState, Tensorizor>::next_instance_id_ = 0;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int Mcts<GameState, Tensorizor>::NNEvaluationService::next_instance_id_ = 0;

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

  boost::filesystem::path default_nnet_filename_path = util::Repo::root() / "c4_model.ptj";
  std::string default_nnet_filename = util::Config::instance()->get(
      "nnet_filename", default_nnet_filename_path.string());

  boost::filesystem::path default_profiling_dir_path = util::Repo::root() / "output" / "mcts_profiling";
  std::string default_profiling_dir = util::Config::instance()->get(
      "mcts_profiling_dir", default_profiling_dir_path.string());

  po2::options_description desc("Mcts options");

  return desc
      .template add_option<"nnet-filename">
          (po::value<std::string>(&nnet_filename)->default_value(default_nnet_filename), "nnet filename")
      .template add_option<"uniform-model">(
          po::bool_switch(&uniform_model), "uniform model (--nnet-filename is ignored)")
      .template add_option<"num-search-threads">(
          po::value<int>(&num_search_threads)->default_value(num_search_threads),
          "num search threads")
      .template add_option<"batch-size-limit">(
          po::value<int>(&batch_size_limit)->default_value(batch_size_limit),
          "batch size limit")
      .template add_bool_switches<"run-offline", "no-run-offline">(
          &run_offline, "run search while opponent is thinking", "do NOT run search while opponent is thinking")
      .template add_option<"offline-tree-size-limit">(
          po::value<int>(&offline_tree_size_limit)->default_value(
          offline_tree_size_limit), "max tree size to grow to offline (only respected in --run-offline mode)")
      .template add_option<"nn-eval-timeout-ns">(
          po::value<int64_t>(&nn_eval_timeout_ns)->default_value(
          nn_eval_timeout_ns), "nn eval thread timeout in ns")
      .template add_option<"cache-size">(
          po::value<size_t>(&cache_size)->default_value(cache_size),
          "nn eval thread cache size")
      .template add_option<"root-softmax-temp">(
          po::value<std::string>(&root_softmax_temperature_str)->default_value(root_softmax_temperature_str),
          "root softmax temperature")
      .template add_option<"cpuct">(po2::float_value("%.2f", &cPUCT), "cPUCT value")
      .template add_option<"dirichlet-mult">(po2::float_value("%.2f", &dirichlet_mult), "dirichlet mult")
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
inline Mcts<GameState, Tensorizor>::NNEvaluation::NNEvaluation(
    const ValueArray1D& value, const PolicyArray1D& policy, const ActionMask& valid_actions)
{
  GameStateTypes::global_to_local(policy, valid_actions, local_policy_logit_distr_);
  value_prob_distr_ = eigen_util::softmax(value);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(Node* p, action_index_t a)
: parent(p)
, action(a)
, tensorizor(p->stable_data().tensorizor)
, state(p->stable_data().state)
, outcome(state.apply_move(action))
{
  tensorizor.receive_state_change(state, action);
  aux_init();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(
    Node* p, action_index_t a, const Tensorizor& t, const GameState& s, const GameOutcome& o)
: parent(p)
, action(a)
, tensorizor(t)
, state(s)
, outcome(o)
{
  aux_init();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(const stable_data_t& data, bool prune_parent)
{
  *this = data;
  if (prune_parent) parent = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::stable_data_t::aux_init() {
  valid_action_mask = state.get_valid_actions();
  current_player = state.get_current_player();
  if (kDeterministic) {
    sym_index = 0;
  } else {
    sym_index = bitset_util::choose_random_on_index(tensorizor.get_symmetry_indices(state));
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::evaluation_data_t::evaluation_data_t(const ActionMask& valid_actions)
: fully_analyzed_actions(~valid_actions) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stats_t::stats_t() {
  value_avg.setZero();
  V_floor.setZero();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(Node* parent, action_index_t action)
: stable_data_(parent, action)
, evaluation_data_(stable_data().valid_action_mask) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome)
: stable_data_(nullptr, -1, tensorizor, state, outcome)
, evaluation_data_(stable_data().valid_action_mask) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(const Node& node, bool prune_parent)
: stable_data_(node.stable_data_, prune_parent)
, children_data_(node.children_data_)
, evaluation_data_(node.evaluation_data_)
, stats_(node.stats_) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline std::string Mcts<GameState, Tensorizor>::Node::genealogy_str() const {
  const char* delim = kNumGlobalActions < 10 ? "" : ":";
  std::vector<std::string> vec;
  const Node* n = this;
  while (n->parent()) {
    vec.push_back(std::to_string(n->action()));
    n = n->parent();
  }

  std::reverse(vec.begin(), vec.end());
  return util::create_string("[%s]", boost::algorithm::join(vec, delim).c_str());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::debug_dump() const {
  std::cout << "value[" << stats_.count << "]: " << stats_.value_avg.transpose() << std::endl;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::release(Node* protected_child) {
  // If we got here, the Node and its children (besides protected_child) should not be referenced from anywhere, so it
  // should be safe to delete it without worrying about thread-safety.
  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (!child) continue;
    if (child != protected_child) child->release();
    clear_child(c);  // not needed currently, but might be needed if we switch to smart-pointers
  }

  delete this;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::adopt_children() {
  // This should only be called in contexts where the search-threads are inactive, so we do not need to worry about
  // thread-safety
  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (child) child->stable_data_.parent = this;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline typename Mcts<GameState, Tensorizor>::GlobalPolicyCountDistr
Mcts<GameState, Tensorizor>::Node::get_effective_counts() const
{
  // This should only be called in contexts where the search-threads are inactive, so we do not need to worry about
  // thread-safety
  bool eliminated = stats_.eliminated();

  player_index_t cp = stable_data().current_player;
  GlobalPolicyCountDistr counts;
  counts.setZero();
  if (eliminated) {
    ValueArray1DExtrema V_floor_extrema = get_V_floor_extrema_among_children();
    float max_V_floor = V_floor_extrema.max[cp];
    for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
      Node* child = get_child(c);
      if (child) {
        counts(child->action()) = (child->stats().V_floor(cp) == max_V_floor);
      }
    }
    return counts;
  }
  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (child) {
      counts(child->action()) = child->stats().effective_count();
    }
  }
  return counts;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::backprop(const ValueProbDistr& outcome)
{
  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.value_avg = (stats_.value_avg * stats_.count + outcome) / (stats_.count + 1);
  stats_.count++;
  lock.unlock();

  if (parent()) parent()->backprop(outcome);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::backprop_with_virtual_undo(
    const ValueProbDistr& value)
{
  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.value_avg += (value - make_virtual_loss()) / stats_.count;
  stats_.virtual_count--;
  lock.unlock();

  if (parent()) parent()->backprop_with_virtual_undo(value);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::virtual_backprop() {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  auto loss = make_virtual_loss();
  stats_.value_avg = (stats_.value_avg * stats_.count + loss) / (stats_.count + 1);
  stats_.count++;
  stats_.virtual_count++;
  lock.unlock();

  if (parent()) parent()->virtual_backprop();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::perform_eliminations(const ValueProbDistr& outcome) {
  ValueArray1DExtrema V_floor_extrema = get_V_floor_extrema_among_children();
  ValueArray1D V_floor;
  player_index_t cp = stable_data().current_player;
  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    if (p == cp) {
      V_floor[p] = V_floor_extrema.max[p];
    } else {
      V_floor[p] = V_floor_extrema.min[p];
    }
  }

  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.V_floor = V_floor;
  bool recurse = parent() && stats_.eliminated();
  lock.unlock();

  if (recurse) {
    parent()->perform_eliminations(outcome);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::ValueArray1D
Mcts<GameState, Tensorizor>::Node::make_virtual_loss() const {
  constexpr float x = 1.0 / (kNumPlayers - 1);
  ValueArray1D virtual_loss;
  virtual_loss.setZero();
  virtual_loss[stable_data().current_player] = x;
  return virtual_loss;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::Node::mark_as_fully_analyzed() {
  Node* my_parent = parent();
  if (!my_parent) return;

  std::unique_lock<std::mutex> lock(my_parent->evaluation_data_mutex());
  my_parent->evaluation_data_.fully_analyzed_actions[action()] = true;
  bool full = my_parent->evaluation_data_.fully_analyzed_actions.all();
  lock.unlock();
  if (!full) return;

  my_parent->mark_as_fully_analyzed();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node* Mcts<GameState, Tensorizor>::Node::init_child(child_index_t c) {
  std::lock_guard guard(children_mutex_);

  Node* child = get_child(c);
  if (child) return child;

  const auto& valid_action_mask = stable_data().valid_action_mask;
  action_index_t action = bitset_util::get_nth_on_index(valid_action_mask, c);

  child = new Node(this, action);
  children_data_.set(c, child);
  return child;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::Node*
Mcts<GameState, Tensorizor>::Node::lookup_child_by_action(action_index_t action) const {
  return get_child(bitset_util::count_on_indices_before(stable_data().valid_action_mask, action));
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::ValueArray1DExtrema
Mcts<GameState, Tensorizor>::Node::get_V_floor_extrema_among_children() const {
  ValueArray1DExtrema extrema;
  extrema.min.setConstant(1);
  extrema.max.setConstant(0);

  for (child_index_t c = 0; c < stable_data_.num_valid_actions(); ++c) {
    Node* child = get_child(c);
    if (child) {
      const ValueArray1D& V_floor = child->stats().V_floor;
      extrema.min = extrema.min.min(V_floor);
      extrema.max = extrema.max.max(V_floor);
    } else {
      extrema.min.setConstant(0);
    }
  }

  return extrema;
}

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
  return mcts_->search_active() && stats.effective_count() <= tree_size_limit && !stats.eliminated();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::visit(Node* tree, int depth) {
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id());
    printer << __func__ << " " << tree->genealogy_str();
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

    if (kEnableThreadingDebug) {
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
inline void Mcts<GameState, Tensorizor>::SearchThread::backprop_outcome(Node* tree, const ValueProbDistr& outcome) {
  record_for_profiling(kBackpropOutcome);
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread_id_);
    printer << __func__ << " " << tree->genealogy_str() << " " << outcome.transpose();
    printer.endl();
  }

  tree->backprop(outcome);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::perform_eliminations(
    Node* tree, const ValueProbDistr& outcome)
{
  if (params_.disable_eliminations) return;
  record_for_profiling(kPerformEliminations);
  tree->perform_eliminations(outcome);
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

  if (kEnableThreadingDebug) {
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
    if (kEnableThreadingDebug) {
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
    ValueArray1D uniform_value;
    PolicyArray1D uniform_policy;
    uniform_value.setConstant(1.0 / kNumPlayers);
    uniform_policy.setConstant(0);
    data->evaluation = std::make_shared<NNEvaluation>(
        uniform_value, uniform_policy, stable_data.valid_action_mask);
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

  LocalPolicyProbDistr P = eigen_util::softmax(data->evaluation->local_policy_logit_distr());
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

  using PVec = LocalPolicyProbDistr;

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

  PUCT *= 1 - E;

  int argmax_index;
  PUCT.maxCoeff(&argmax_index);

  mcts_->record_puct_calc(VN.sum() > 0);

  if (kEnableThreadingDebug) {
    std::string genealogy = tree->genealogy_str();

    util::ThreadSafePrinter printer(thread_id());

    printer << "*************";
    printer.endl();
    printer << __func__ << "() " << genealogy;
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

  std::bitset<kNumGlobalActions> fpu_bits;
  fpu_bits.set();

  for (child_index_t c = 0; c < tree->stable_data().num_valid_actions(); ++c) {
    /*
     * NOTE: we do NOT grab the child stats_mutex here! This means that child_stats can contain
     * arbitrarily-partially-written data.
     */
    Node* child = tree->get_child(c);
    if (!child) continue;
    auto child_stats = child->stats();  // struct copy to simplify reasoning about race conditions

    V(c) = child_stats.effective_value_avg(cp);
    N(c) = child_stats.effective_count();
    VN(c) = child_stats.virtual_count;
    E(c) = child_stats.eliminated();

    fpu_bits[c] = (N(c) == 0);
  }

  if (params.enable_first_play_urgency && fpu_bits.any()) {
    /*
     * Again, we do NOT grab the stats_mutex here!
     */
    const auto& stats = tree->stats();  // no struct copy, not needed here
    dtype PV = stats.effective_value_avg(cp);

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
    const boost::filesystem::path& net_filename, int batch_size, std::chrono::nanoseconds timeout_duration,
    size_t cache_size, const boost::filesystem::path& profiling_dir)
: instance_id_(next_instance_id_++)
, net_(net_filename)
, batch_data_(batch_size)
, cache_(cache_size)
, timeout_duration_(timeout_duration)
, batch_size_limit_(batch_size)
{
  torch_input_gpu_ = batch_data_.input.asTorch().clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);
  deadline_ = std::chrono::steady_clock::now();

  std::string name = util::create_string("eval-%d", instance_id_);
  auto profiling_filename = profiling_dir / util::create_string("%s.txt", name.c_str());
  init_profiling(profiling_filename.c_str(), name.c_str());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationService::batch_data_t::batch_data_t(
    int batch_size)
: policy(batch_size, kNumGlobalActions, util::to_std_array<int>(batch_size, kNumGlobalActions))
, value(batch_size, kNumPlayers, util::to_std_array<int>(batch_size, kNumPlayers))
, input(util::to_std_array<int>(batch_size, util::std_array_v<int, typename Tensorizor::Shape>))
{
  input.asEigen().setZero();
  eval_ptr_data = new eval_ptr_data_t[batch_size];
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationService::~NNEvaluationService() {
  disconnect();
  delete[] batch_data_.eval_ptr_data;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::NNEvaluationService::Response
Mcts<GameState, Tensorizor>::NNEvaluationService::evaluate(const Request& request) {
  SearchThread* thread = request.thread;
  const Node* tree = request.tree;

  if (kEnableThreadingDebug) {
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

  if (kEnableThreadingDebug) {
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
float Mcts<GameState, Tensorizor>::NNEvaluationService::global_avg_batch_size() {
  int64_t num = 0;
  int64_t den = 0;

  for (auto it : instance_map_) {
    NNEvaluationService* service = it.second;
    num += service->evaluated_positions();
    den += service->batches_evaluated();
  }

  return num * 1.0 / std::max(int64_t(1), den);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::batch_evaluate() {
  std::unique_lock batch_metadata_lock(batch_metadata_.mutex);
  std::unique_lock batch_data_lock(batch_data_.mutex);

  assert(batch_metadata_.reserve_index > 0);
  assert(batch_metadata_.reserve_index == batch_metadata_.commit_count);

  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- NNEvaluationService::%s(%s) ---------------------->\n",
                   __func__, batch_metadata_.repr().c_str());
  }

  record_for_profiling(kCopyingCpuToGpu);
  torch_input_gpu_.copy_(batch_data_.input.asTorch());
  record_for_profiling(kEvaluatingNeuralNet);
  net_.predict(input_vec_, batch_data_.policy.asTorch(), batch_data_.value.asTorch());

  record_for_profiling(kCopyingToPool);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    eval_ptr_data_t &edata = batch_data_.eval_ptr_data[i];
    auto &policy = batch_data_.policy.eigenSlab(i);
    auto &value = batch_data_.value.eigenSlab(i);

    edata.transform->transform_policy(policy);
    edata.eval_ptr.store(std::make_shared<NNEvaluation>(
        eigen_util::to_array1d(value), eigen_util::to_array1d(policy), edata.valid_actions));
  }

  record_for_profiling(kAcquiringCacheMutex);
  std::unique_lock<std::mutex> lock(cache_mutex_);
  record_for_profiling(kFinishingUp);
  for (int i = 0; i < batch_metadata_.reserve_index; ++i) {
    const eval_ptr_data_t &edata = batch_data_.eval_ptr_data[i];
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

  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  waiting for cache lock...\n");
  }

  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  auto cached = cache_.get(cache_key);
  if (cached.has_value()) {
    if (kEnableThreadingDebug) {
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
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&]{
    if (batch_metadata_.unread_count == 0 && batch_metadata_.reserve_index < batch_size_limit_ && batch_metadata_.accepting_reservations) return true;
    if (kEnableThreadingDebug) {
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
  if (kEnableThreadingDebug) {
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
   * The significance of not yet being committed is that the service thread won't yet proceed with nnet eval.
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
  symmetry_index_t sym_index = cache_key.sym_index;

  thread->record_for_profiling(SearchThread::kTensorizing);
  std::unique_lock<std::mutex> lock(batch_data_.mutex);

  auto& input = batch_data_.input.template eigenSlab<typename TensorizorTypes::Shape<1>>(reserve_index);
  tensorizor.tensorize(input, state);
  auto transform = tensorizor.get_symmetry(sym_index);
  transform->transform_input(input);

  batch_data_.eval_ptr_data[reserve_index].eval_ptr.store(nullptr);
  batch_data_.eval_ptr_data[reserve_index].cache_key = cache_key;
  batch_data_.eval_ptr_data[reserve_index].valid_actions = valid_action_mask;
  batch_data_.eval_ptr_data[reserve_index].transform = transform;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::increment_commit_count(SearchThread* thread)
{
  thread->record_for_profiling(SearchThread::kIncrementingCommitCount);

  batch_metadata_.commit_count++;
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", __func__, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.notify_one();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
Mcts<GameState, Tensorizor>::NNEvaluation_sptr Mcts<GameState, Tensorizor>::NNEvaluationService::get_eval(
    SearchThread* thread, int reserve_index, std::unique_lock<std::mutex>& metadata_lock)
{
  const char* func = __func__;
  thread->record_for_profiling(SearchThread::kWaitingForReservationProcessing);
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&]{
    if (batch_metadata_.reserve_index == 0) return true;
    if (kEnableThreadingDebug) {
      util::ThreadSafePrinter printer(thread->thread_id());
      printer.printf("  %s(%s) still waiting...\n", func, batch_metadata_.repr().c_str());
    }
    return false;
  });

  return batch_data_.eval_ptr_data[reserve_index].eval_ptr.load();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::wait_until_all_read(
    SearchThread* thread, std::unique_lock<std::mutex>& metadata_lock)
{
  assert(batch_metadata_.unread_count > 0);
  batch_metadata_.unread_count--;

  const char* func = __func__;
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer(thread->thread_id());
    printer.printf("  %s(%s)...\n", func, batch_metadata_.repr().c_str());
  }
  cv_evaluate_.wait(metadata_lock, [&]{
    if (batch_metadata_.unread_count == 0) return true;
    if (kEnableThreadingDebug) {
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
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&]{
    if (batch_metadata_.unread_count == 0) return true;
    if (kEnableThreadingDebug) {
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
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&]{
    if (batch_metadata_.reserve_index > 0) return true;
    if (kEnableThreadingDebug) {
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
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait_until(lock, deadline_, [&]{
    if (batch_metadata_.reserve_index == batch_size_limit_) return true;
    if (kEnableThreadingDebug) {
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
  if (kEnableThreadingDebug) {
    util::ThreadSafePrinter printer;
    printer.printf("<---------------------- %s %s(%s) ---------------------->\n",
                   cls, func, batch_metadata_.repr().c_str());
  }
  cv_service_loop_.wait(lock, [&]{
    if (batch_metadata_.reserve_index == batch_metadata_.commit_count) return true;
    if (kEnableThreadingDebug) {
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
, offline_search_params_(SearchParams::make_offline_params(params.offline_tree_size_limit))
, instance_id_(next_instance_id_++)
, root_softmax_temperature_(math::ExponentialDecay::parse(
    params.root_softmax_temperature_str, GameStateTypes::get_var_bindings()))
{
  namespace bf = boost::filesystem;

  if (kEnableProfiling) {
    if (profiling_dir().empty()) {
      throw util::Exception("Required: --mcts-profiling-dir. Alternatively, add entry for 'mcts_profiling_dir' in config.txt");
    }
    init_profiling_dir(profiling_dir().string());
  }

  if (!params.uniform_model) {
    nn_eval_service_ = NNEvaluationService::create(this);
  }
  if (num_search_threads() < 1) {
    throw util::Exception("num_search_threads must be positive (%d)", num_search_threads());
  }
  if (params.run_offline && num_search_threads() == 1) {
    throw util::Exception("run_offline does not work with only 1 search thread");
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
    player_index_t player, const GameState& state, action_index_t action)
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

  if (params_.run_offline) {
    start_search_threads(&offline_search_params_);
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
  results_.counts = root_->get_effective_counts().template cast<dtype>();
  if (params_.forced_playouts && add_noise) {
    prune_counts(params);
  }
  results_.policy_prior = evaluation_data.local_policy_prob_distr;
  results_.win_rates = root_->stats().value_avg;
  results_.value_prior = evaluation->value_prob_distr();
  return &results_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::add_dirichlet_noise(LocalPolicyProbDistr& P) {
  int rows = P.rows();
  double alpha = params_.dirichlet_alpha_sum / rows;
  LocalPolicyProbDistr noise = dirichlet_gen_.template generate<LocalPolicyProbDistr>(
      rng_, alpha, rows);
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
  /*
   * Thread-safety analysis:
   *
   * - changes in root_ are always synchronized via stop_search_threads()
   * - race-conditions on root_->stats_ reads can at worst cause us to do more visits than required
   */
  while (thread->needs_more_visits(root_, tree_size_limit)) {
    thread->visit(root_, 1);
    thread->dump_profiling_stats();
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
  if (params_.uniform_model) return;

  PUCTStats stats(params_, search_params, root_);

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

  if (!results_.counts.isFinite().all() || results_.counts.sum() <= 0) {
    std::cout << "P: " << P.transpose() << std::endl;
    std::cout << "N: " << N.transpose() << std::endl;
    std::cout << "V: " << V.transpose() << std::endl;
    std::cout << "PUCT: " << PUCT.transpose() << std::endl;
    std::cout << "n_forced: " << n_forced.transpose() << std::endl;
    std::cout << "results_.counts: " << results_.counts.transpose() << std::endl;
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

}  // namespace common
