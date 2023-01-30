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
int Mcts<GameState, Tensorizor>::next_instance_id_ = 0;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
int Mcts<GameState, Tensorizor>::NNEvaluationService::next_instance_id_ = 0;

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
template<boost_util::program_options::OptionStyle Style>
auto Mcts<GameState, Tensorizor>::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  boost::filesystem::path default_nnet_filename_path = util::Repo::root() / "c4_model.ptj";
  std::string default_nnet_filename = util::Config::instance()->get(
      "nnet_filename", default_nnet_filename_path.string());

  boost::filesystem::path default_profiling_dir_path = util::Repo::root() / "output" / "mcts_profiling";
  std::string default_profiling_dir = util::Config::instance()->get(
      "mcts_profiling_dir", default_profiling_dir_path.string());

  po2::options_description<Style> desc("Mcts options");

  return desc
      .template add_option<"nnet-filename">
          (po::value<std::string>(&nnet_filename)->default_value(default_nnet_filename), "nnet filename")
      .template add_option<"num-search-threads">(
          po::value<int>(&num_search_threads)->default_value(num_search_threads),
          "num search threads")
      .template add_option<"batch-size-limit">(
          po::value<int>(&batch_size_limit)->default_value(batch_size_limit),
          "batch size limit")
      .template add_option<"run-offline">(
          po2::store_bool(&run_offline, true),
          po2::make_store_bool_help_str("run search while opponent is thinking", run_offline).c_str())
      .template add_option<"no-run-offline">(
          po2::store_bool(&run_offline, false),
          po2::make_store_bool_help_str("do NOT run search while opponent is thinking", !run_offline).c_str())
      .template add_option<"offline-tree-size-limit">(
          po::value<int>(&offline_tree_size_limit)->default_value(
          offline_tree_size_limit), "max tree size to grow to offline (only respected in --run-offline mode)")
      .template add_option<"nn-eval-timeout-ns">(
          po::value<int64_t>(&nn_eval_timeout_ns)->default_value(
          nn_eval_timeout_ns), "nn eval thread timeout in ns")
      .template add_option<"cache-size">(
          po::value<size_t>(&cache_size)->default_value(cache_size),
          "nn eval thread cache size")
      .template add_option<"root-softmax-temp">(po2::float_value("%.2f", &root_softmax_temperature), "root softmax temperature")
      .template add_option<"cpuct">(po2::float_value("%.2f", &cPUCT), "cPUCT value")
      .template add_option<"dirichlet-mult">(po2::float_value("%.2f", &dirichlet_mult), "dirichlet mult")
      .template add_option<"dirichlet-alpha">(po2::float_value("%.2f", &dirichlet_alpha), "dirichlet alpha")
      .template add_option<"disable-eliminations">(po2::store_bool(&disable_eliminations, true),
          po2::make_store_bool_help_str("disable eliminations", disable_eliminations).c_str())
      .template add_option<"no-disable-eliminations">(po2::store_bool(&disable_eliminations, false),
          po2::make_store_bool_help_str("enable eliminations", !disable_eliminations).c_str())
      .template add_option<"speculative-evals">(po2::store_bool(&speculative_evals, true),
          po2::make_store_bool_help_str("enable speculation", speculative_evals).c_str())
      .template add_option<"no-speculative-evals">(po2::store_bool(&speculative_evals, false),
          po2::make_store_bool_help_str("disable speculation", !speculative_evals).c_str())
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
inline Mcts<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(
    Node* parent, action_index_t action, bool disable_noise)
: parent_(parent)
, action_(action)
, disable_noise_(disable_noise) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stable_data_t::stable_data_t(const stable_data_t& data, bool prune_parent)
{
  *this = data;
  if (prune_parent) parent_ = nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::lazily_initialized_data_t::data_t::data_t(
    Node* parent, action_index_t action)
: tensorizor_(parent->_tensorizor())
, state_(parent->_state())
{
  outcome_ = state_.apply_move(action);
  valid_action_mask_ = state_.get_valid_actions();
  current_player_ = state_.get_current_player();
  sym_index_ = bitset_util::choose_random_on_index(tensorizor_.get_symmetry_indices(state_));
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::lazily_initialized_data_t::data_t::data_t(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome)
: tensorizor_(tensorizor)
, state_(state)
, outcome_(outcome)
{
  valid_action_mask_ = state_.get_valid_actions();
  current_player_ = state_.get_current_player();
  sym_index_ = bitset_util::choose_random_on_index(tensorizor_.get_symmetry_indices(state_));
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::evaluation_data_t::evaluation_data_t(const ActionMask& valid_actions)
: fully_analyzed_actions_(~valid_actions) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::stats_t::stats_t() {
  value_avg_.setZero();
  effective_value_avg_.setZero();
  V_floor_.setZero();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(Node* parent, action_index_t action)
: stable_data_(parent, action, true) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(
    const Tensorizor& tensorizor, const GameState& state, const GameOutcome& outcome, bool disable_noise)
: stable_data_(nullptr, -1, disable_noise)
, lazily_initialized_data_(tensorizor, state, outcome)
, evaluation_data_(_valid_action_mask()) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::Node::Node(const Node& node, bool prune_parent)
: stable_data_(node.stable_data_, prune_parent)
, lazily_initialized_data_(node.lazily_initialized_data_)
, children_data_(node.children_data_)
, evaluation_data_(node.evaluation_data_)
, stats_(node.stats_) {}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::debug_dump() const {
  std::cout << "value[" << stats_.count_ << "]: " << stats_.value_avg_.transpose() << std::endl;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::_release(Node* protected_child) {
  Node* first_child;
  int num_children;
  children_data_.read(&first_child, &num_children);

  for (int i = 0; i < num_children; ++i) {
    Node* child = first_child + i;
    if (child != protected_child) child->_release();
  }

  if (!first_child) return;

  // https://stackoverflow.com/a/4756306/543913
  for (int i = num_children - 1; i >= 0; --i) {
    Node* child = first_child + i;
    child->~Node();
  }
  void* raw_memory = static_cast<void*>(first_child);
  operator delete[](raw_memory);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::_adopt_children() {
  Node* first_child;
  int num_children;
  children_data_.read(&first_child, &num_children);

  for (int i = 0; i < num_children; ++i) {
    Node* child = first_child + i;
    child->stable_data_.parent_ = this;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline typename Mcts<GameState, Tensorizor>::GlobalPolicyCountDistr
Mcts<GameState, Tensorizor>::Node::get_effective_counts() const
{
  std::unique_lock<std::mutex> lock(stats_mutex_);
  bool eliminated = stats_.eliminated_;
  lock.unlock();

  Node* first_child;
  int num_children;
  children_data_.read(&first_child, &num_children);

  player_index_t cp = _current_player();
  GlobalPolicyCountDistr counts;
  counts.setZero();
  if (eliminated) {
    float max_V_floor = _get_max_V_floor_among_children(cp, first_child, num_children);
    for (int i = 0; i < num_children; ++i) {
      Node* child = first_child + i;
      counts(child->action()) = (child->_V_floor(cp) == max_V_floor);
    }
    return counts;
  }
  for (int i = 0; i < num_children; ++i) {
    Node* child = first_child + i;
    counts(child->action()) = child->_effective_count();
  }
  return counts;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::backprop(const ValueProbDistr& outcome)
{
  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.value_avg_ = (stats_.value_avg_ * stats_.count_ + outcome) / (stats_.count_ + 1);
  stats_.count_++;
  stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  lock.unlock();

  if (parent()) parent()->backprop(outcome);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::backprop_with_virtual_undo(
    const ValueProbDistr& value)
{
  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.value_avg_ += (value - make_virtual_loss()) / stats_.count_;
  stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  lock.unlock();

  if (parent()) parent()->backprop_with_virtual_undo(value);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::virtual_backprop() {
  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.value_avg_ = (stats_.value_avg_ * stats_.count_ + make_virtual_loss()) / (stats_.count_ + 1);
  stats_.count_++;
  stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  lock.unlock();

  if (parent()) parent()->virtual_backprop();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::perform_eliminations(const ValueProbDistr& outcome) {
  Node* first_child;
  int num_children;
  children_data_.read(&first_child, &num_children);

  ValueArray1D V_floor;
  player_index_t cp = _current_player();
  for (player_index_t p = 0; p < kNumPlayers; ++p) {
    if (p == cp) {
      V_floor[p] = _get_max_V_floor_among_children(p, first_child, num_children);
    } else {
      V_floor[p] = _get_min_V_floor_among_children(p, first_child, num_children);
    }
  }

  bool recurse = false;

  std::unique_lock<std::mutex> lock(stats_mutex_);
  stats_.V_floor_ = V_floor;
  stats_.effective_value_avg_ = _has_certain_outcome() ? stats_.V_floor_ : stats_.value_avg_;
  if (_can_be_eliminated()) {
    stats_.eliminated_ = true;
    recurse = parent();
  }
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
  virtual_loss = x;
  virtual_loss[_current_player()] = 0;
  return virtual_loss;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::Node::mark_as_fully_analyzed() {
  Node* my_parent = parent();
  if (!my_parent) return;

  std::unique_lock<std::mutex> lock(my_parent->evaluation_data_mutex());
  my_parent->evaluation_data_.fully_analyzed_actions_[action()] = true;
  bool full = my_parent->evaluation_data_.fully_analyzed_actions_.all();
  lock.unlock();
  if (!full) return;

  my_parent->mark_as_fully_analyzed();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::Node::_lazy_init() {
  new(&lazily_initialized_data_) lazily_initialized_data_t(parent(), action());

  std::unique_lock<std::mutex> lock(evaluation_data_mutex());
  new(&evaluation_data_) evaluation_data_t(_valid_action_mask());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::Node::_expand_children() {
  int num_children = _valid_action_mask().count();
  void* raw_memory = operator new[](num_children * sizeof(Node));
  Node* node = static_cast<Node*>(raw_memory);

  Node* first_child = node;
  for (action_index_t action : bitset_util::on_indices(_valid_action_mask())) {
    new(node++) Node(this, action);
  }
  children_data_.write(first_child, num_children);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::Node*
Mcts<GameState, Tensorizor>::Node::_find_child(action_index_t action) const {
  // TODO: technically we can do a binary search here, as children should be in sorted order by action
  Node* first_child;
  int num_children;
  children_data_.read(&first_child, &num_children);

  for (int i = 0; i < num_children; ++i) {
    Node *child = first_child + i;
    if (child->stable_data_.action_ == action) return child;
  }
  return nullptr;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts<GameState, Tensorizor>::Node::_get_max_V_floor_among_children(
    player_index_t p, Node* first_child, int num_children) const
{
  float max_V_floor = 0;
  for (int i = 0; i < num_children; ++i) {
    Node* child = first_child + i;
    std::lock_guard<std::mutex> guard(child->stats_mutex_);
    max_V_floor = std::max(max_V_floor, child->_V_floor(p));
  }
  return max_V_floor;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline float Mcts<GameState, Tensorizor>::Node::_get_min_V_floor_among_children(
    player_index_t p, Node* first_child, int num_children) const
{
  float min_V_floor = 1;
  for (int i = 0; i < num_children; ++i) {
    Node* child = first_child + i;
    std::lock_guard<std::mutex> guard(child->stats_mutex_);
    min_V_floor = std::min(min_V_floor, child->_V_floor(p));
  }
  return min_V_floor;
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
inline void Mcts<GameState, Tensorizor>::SearchThread::launch(int tree_size_limit) {
  kill();
  thread_ = new std::thread([&] { mcts_->run_search(this, tree_size_limit); });
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
bool Mcts<GameState, Tensorizor>::SearchThread::needs_more_visits(Node* root, int tree_size_limit) {
  record_for_profiling(kCheckVisitReady);
  return mcts_->search_active() && root->_effective_count() <= tree_size_limit && !root->_eliminated();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::visit(Node* tree, int depth) {
  lazily_init(tree);
  const auto& outcome = tree->_outcome();
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
  assert(evaluation == tree->_evaluation().get());

  Node* best_child = get_best_child(tree, evaluation);
  if (data.performed_expansion) {
    record_for_profiling(kBackpropEvaluation);
    tree->backprop_with_virtual_undo(evaluation->value_prob_distr());
  } else {
    visit(best_child, depth + 1);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::lazily_init(Node* tree) {
  record_for_profiling(kAcquiringLazilyInitializedDataMutex);
  std::lock_guard<std::mutex> guard(tree->lazily_initialized_data_mutex());
  if (tree->_lazily_initialized()) return;
  record_for_profiling(kLazyInit);
  tree->_lazy_init();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::SearchThread::backprop_outcome(Node* tree, const ValueProbDistr& outcome) {
  record_for_profiling(kBackpropOutcome);
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
  evaluate_and_expand_result_t data{tree->_evaluation(), false};
  typename Node::evaluation_state_t state = tree->_evaluation_state();

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
        if (tree->_evaluation_state() != Node::kSet) {
          tree->cv_evaluate_and_expand().wait(lock);
          assert(tree->_evaluation_state() == Node::kSet);
        }
        data.evaluation = tree->_evaluation();
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

  assert(!tree->_has_children());
  expand_children(tree);
  data->performed_expansion = true;
  assert(data->evaluation.get() == nullptr);

  if (params_.speculative_evals) {
    tree->_set_evaluation_state(Node::kPending);
    lock->unlock();
  }

  if (!speculative) {
    record_for_profiling(kVirtualBackprop);
    tree->virtual_backprop();
  }

  symmetry_index_t sym_index = tree->_sym_index();
  typename NNEvaluationService::Request request{
      this, &tree->_tensorizor(), &tree->_state(), &tree->_valid_action_mask(), sym_index
  };
  auto response = mcts_->nn_eval_service()->evaluate(request);
  data->evaluation = response.ptr;

  if (params_.speculative_evals) {
    if (speculative && response.used_cache) {
      // without this, when we hit cache, we fail to saturate nn service batch
      lock->lock();
      evaluate_and_expand_pending(tree, lock);
    }

    lock->lock();
  }
  float inv_temp = tree->is_root() ? (1.0 / params_.root_softmax_temperature) : 1.0;
  tree->_set_local_policy_prob_distr(eigen_util::softmax(data->evaluation->local_policy_logit_distr() * inv_temp));
  tree->_set_evaluation(data->evaluation);
  tree->_set_evaluation_state(Node::kSet);
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::SearchThread::evaluate_and_expand_pending(
    Node* tree, std::unique_lock<std::mutex>* lock)
{
  // Another search thread is working on this. Might as well speculatively eval another position while we wait
  record_for_profiling(kEvaluateAndExpandPending);

  assert(tree->_has_children());
  Node* child;
  if (tree->_fully_analyzed_action_mask().all()) {
    child = tree->_get_child(0);
    lock->unlock();
  } else {
    action_index_t action = bitset_util::choose_random_off_index(tree->_fully_analyzed_action_mask());
    lock->unlock();
    child = tree->_find_child(action);
  }
  lazily_init(child);
  const auto& outcome = child->_outcome();
  if (is_terminal_outcome(outcome)) {
    perform_eliminations(child, outcome);  // why not?
    mark_as_fully_analyzed(child);
  } else {
    evaluate_and_expand(child, true);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::SearchThread::expand_children(Node* tree) {
  if (tree->_has_children()) return;

  // TODO: use object pool
  record_for_profiling(kConstructingChildren);
  tree->_expand_children();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::Node*
Mcts<GameState, Tensorizor>::SearchThread::get_best_child(
    Node* tree, NNEvaluation* evaluation)
{
  record_for_profiling(kPUCT);

  using PVec = LocalPolicyProbDistr;

  PVec P = tree->_local_policy_prob_distr();
  if (tree->is_root() && !tree->disable_noise() && params_.dirichlet_mult) {
    mcts_->add_dirichlet_noise(P);
  }

  const int rows = P.rows();
  assert(rows == int(tree->_valid_action_mask().count()));

  PVec V(rows);
  PVec N(rows);
  PVec E(rows);
  player_index_t cp = tree->_current_player();

  for (int c = 0; c < tree->_num_children(); ++c) {
    Node* child = tree->_get_child(c);
    record_for_profiling(kAcquiringStatsMutex);
    std::lock_guard<std::mutex> guard(child->stats_mutex());
    record_for_profiling(kPUCT);

    V(c) = child->_effective_value_avg(cp);
    N(c) = child->_effective_count();
    E(c) = child->_eliminated();
  }

  constexpr float eps = 1e-6;  // needed when N == 0
  PVec PUCT = V + params_.cPUCT * P * sqrt(N.sum() + eps) / (N + 1);
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
  return best_child;
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
, policy_batch_(batch_size, kNumGlobalActions, util::to_std_array<int>(batch_size, kNumGlobalActions))
, value_batch_(batch_size, kNumPlayers, util::to_std_array<int>(batch_size, kNumPlayers))
, input_batch_(util::to_std_array<int>(batch_size, util::std_array_v<int, typename Tensorizor::Shape>))
, cache_(cache_size)
, timeout_duration_(timeout_duration)
, batch_size_limit_(batch_size)
{
  evaluation_data_batch_ = new evaluation_data_t[batch_size];
  torch_input_gpu_ = input_batch_.asTorch().clone().to(torch::kCUDA);
  input_vec_.push_back(torch_input_gpu_);
  deadline_ = std::chrono::steady_clock::now();

  std::string name = util::create_string("eval-%d", instance_id_);
  auto profiling_filename = profiling_dir / util::create_string("%s.txt", name.c_str());
  init_profiling(profiling_filename.c_str(), name.c_str());
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline Mcts<GameState, Tensorizor>::NNEvaluationService::~NNEvaluationService() {
  disconnect();
  delete[] evaluation_data_batch_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
typename Mcts<GameState, Tensorizor>::NNEvaluationService::Response
Mcts<GameState, Tensorizor>::NNEvaluationService::evaluate(const Request& request) {
  SearchThread* thread = request.thread;
  const Tensorizor& tensorizor = *request.tensorizor;
  const GameState& state = *request.state;
  const ActionMask& valid_action_mask = *request.valid_action_mask;
  symmetry_index_t sym_index = request.sym_index;

  thread->record_for_profiling(SearchThread::kCheckingCache);
  cache_key_t key{state, sym_index};

  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  auto cached = cache_.get(key);
  if (cached.has_value()) {
    cache_hits_++;
    return Response{cached.value(), true};
  }
  cache_lock.unlock();
  cache_misses_++;

  thread->record_for_profiling(SearchThread::kAcquiringBatchMutex);
  std::unique_lock<std::mutex> batch_lock(batch_mutex_);

  thread->record_for_profiling(SearchThread::kWaitingUntilBatchReservable);
  cv_evaluate_.wait(batch_lock, [&]{ return batch_reservable(); });

  thread->record_for_profiling(SearchThread::kMisc);
  int my_index = batch_reserve_index_;
  assert(my_index < batch_size_limit_);
  batch_reserve_index_++;
  if (my_index == 0) {
    deadline_ = std::chrono::steady_clock::now() + timeout_duration_;
  }
  assert(batch_commit_count_ < batch_reserve_index_);
  batch_lock.unlock();
  cv_service_loop_.notify_one();

  /*
   * At this point, the work unit is effectively RESERVED but not COMMITTED.
   *
   * The significance of being reserved is that other search threads will be blocked from reserving if the batch is
   * fully reserved.
   *
   * The significance of not yet being committed is that the service thread won't yet proceed with nnet eval.
   */
  thread->record_for_profiling(SearchThread::kTensorizing);

  auto& input = input_batch_.template eigenSlab<typename TensorizorTypes::Shape<1>>(my_index);
  tensorizor.tensorize(input, state);
  auto transform = tensorizor.get_symmetry(sym_index);
  transform->transform_input(input);

  evaluation_data_batch_[my_index].eval_ptr.store(nullptr);
  evaluation_data_batch_[my_index].cache_key = key;
  evaluation_data_batch_[my_index].valid_actions = valid_action_mask;
  evaluation_data_batch_[my_index].transform = transform;

  thread->record_for_profiling(SearchThread::kIncrementingCommitCount);
  std::unique_lock<std::mutex> lock(batch_mutex_);
  batch_commit_count_++;
  lock.unlock();
  cv_service_loop_.notify_one();

  thread->record_for_profiling(SearchThread::kAcquiringBatchMutex);
  lock.lock();
  thread->record_for_profiling(SearchThread::kWaitingForReservationProcessing);
  cv_evaluate_.wait(lock, [&]{ return batch_reservations_empty(); });

  NNEvaluation_sptr eval_ptr = evaluation_data_batch_[my_index].eval_ptr.load();
  assert(batch_unread_count_ > 0);
  batch_unread_count_--;
  lock.unlock();

  // NOTE: might be able to notify_one(), if we add another notify_one() after the batch_reserve_index_++
  cv_evaluate_.notify_all();
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
  assert(batch_reserve_index_ > 0);
  assert(batch_reserve_index_ == batch_commit_count_);

  record_for_profiling(kCopyingCpuToGpu);
  torch_input_gpu_.copy_(input_batch_.asTorch());
  record_for_profiling(kEvaluatingNeuralNet);
  net_.predict(input_vec_, policy_batch_.asTorch(), value_batch_.asTorch());

  record_for_profiling(kCopyingToPool);
  for (int i = 0; i < batch_reserve_index_; ++i) {
    evaluation_data_t &edata = evaluation_data_batch_[i];
    auto &policy = policy_batch_.eigenSlab(i);
    auto &value = value_batch_.eigenSlab(i);

    edata.transform->transform_policy(policy);
    edata.eval_ptr.store(std::make_shared<NNEvaluation>(
        eigen_util::to_array1d(value), eigen_util::to_array1d(policy), edata.valid_actions));
  }

  record_for_profiling(kAcquiringCacheMutex);
  std::unique_lock<std::mutex> lock(cache_mutex_);
  record_for_profiling(kFinishingUp);
  for (int i = 0; i < batch_reserve_index_; ++i) {
    const evaluation_data_t &edata = evaluation_data_batch_[i];
    cache_.insert(edata.cache_key, edata.eval_ptr);
  }
  lock.unlock();

  evaluated_positions_ += batch_reserve_index_;
  batches_evaluated_++;

  batch_unread_count_ = batch_commit_count_;
  batch_reserve_index_ = 0;
  batch_commit_count_ = 0;
  cv_evaluate_.notify_all();
  dump_profiling_stats();
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NNEvaluationService::loop() {
  while (active()) {
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
//  printf("NodeReleaseService release_count=%d max_queue_size=%d\n", release_count_, max_queue_size_);
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
      unit.node->_release(unit.arg);
      delete unit.node;
    }
    queue.clear();
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
void Mcts<GameState, Tensorizor>::NodeReleaseService::_release(Node* node, Node* arg) {
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
, instance_id_(next_instance_id_++)
{
  namespace bf = boost::filesystem;

  if (kEnableProfiling) {
    if (profiling_dir().empty()) {
      throw util::Exception("Required: --mcts-profiling-dir. Alternatively, add entry for 'mcts_profiling_dir' in config.txt");
    }
    init_profiling_dir(profiling_dir().string());
  }

  nn_eval_service_ = NNEvaluationService::create(this);
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
  nn_eval_service_->disconnect();
  for (auto* thread : search_threads_) {
    delete thread;
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::start() {
  clear();

  if (!connected_) {
    nn_eval_service_->connect();
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
    player_index_t player, const GameState& state, action_index_t action, const GameOutcome& outcome)
{
  stop_search_threads();
  if (!root_) return;

  assert(root_->parent()==nullptr);

  Node* new_root = root_->_find_child(action);
  if (!new_root) {
    NodeReleaseService::release(root_);
    root_ = nullptr;
    return;
  }

  if (!new_root->_lazily_initialized()) {
    new_root->_lazy_init();
  }
  Node* new_root_copy = new Node(*new_root, true);
  NodeReleaseService::release(root_, new_root);
  root_ = new_root_copy;
  root_->_adopt_children();

  if (params_.run_offline) {
    start_search_threads(params_.offline_tree_size_limit);
  }
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline const typename Mcts<GameState, Tensorizor>::MctsResults* Mcts<GameState, Tensorizor>::sim(
    const Tensorizor& tensorizor, const GameState& game_state, const SimParams& params)
{
  stop_search_threads();

  if (!root_ || (!params.disable_noise && params_.dirichlet_mult > 0)) {
    if (root_) {
      NodeReleaseService::release(root_);
    }
    auto outcome = make_non_terminal_outcome<kNumPlayers>();
    root_ = new Node(tensorizor, game_state, outcome, params.disable_noise);  // TODO: use memory pool
  }

  start_search_threads(params.tree_size_limit);
  wait_for_search_threads();

  NNEvaluation_sptr evaluation = root_->_evaluation();
  results_.valid_actions = root_->_valid_action_mask();
  results_.counts = root_->get_effective_counts();
  results_.policy_prior = root_->_local_policy_prob_distr();
  results_.win_rates = root_->_value_avg();
  results_.value_prior = evaluation->value_prob_distr();
  return &results_;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::add_dirichlet_noise(LocalPolicyProbDistr& P) {
  int rows = P.rows();
  LocalPolicyProbDistr noise = dirichlet_gen_.template generate<LocalPolicyProbDistr>(
      rng_, params_.dirichlet_alpha, rows);
  P = (1.0 - params_.dirichlet_mult) * P + params_.dirichlet_mult * noise;
}

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
inline void Mcts<GameState, Tensorizor>::start_search_threads(int tree_size_limit) {
  assert(!search_active_);
  search_active_ = true;
  num_active_search_threads_ = num_search_threads();

  for (auto* thread : search_threads_) {
    thread->launch(tree_size_limit);
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
   * - _effective_count() and _eliminated() read root_->stats_.{eliminated_, count_}
   *   - eliminated_ starts false and gets flipped to true at most once.
   *   - count_ is monotonoically increasing
   *   - Race-conditions can lead us to read stale values of these. That is ok - that merely causes us to possibly to
   *     more visits than a thread-safe alternative would do.
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
