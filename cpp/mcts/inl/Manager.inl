#include <mcts/Manager.hpp>

#include <boost/filesystem.hpp>

#include <util/Exception.hpp>

namespace mcts {

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
int Manager<GameState, Tensorizor>::next_instance_id_ = 0;

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Manager<GameState, Tensorizor>::Manager(const ManagerParams& params)
: params_(params)
, pondering_search_params_(SearchParams::make_pondering_params(params.pondering_tree_size_limit))
{
  shared_data_.manager_id = next_instance_id_++;
  new (&shared_data_.root_softmax_temperature) math::ExponentialDecay(math::ExponentialDecay::parse(
      params.root_softmax_temperature_str, GameStateTypes::get_var_bindings()));
  namespace bf = boost::filesystem;

  if (mcts::kEnableProfiling) {
    auto profiling_dir = params_.profiling_dir();
    if (profiling_dir.empty()) {
      throw util::Exception("Required: --mcts-profiling-dir. Alternatively, add entry for 'mcts_profiling_dir' in config.txt");
    }
    init_profiling_dir(profiling_dir.string());
  }

  if (!params.model_filename.empty()) {
    nn_eval_service_ = NNEvaluationService::create(params);
  }
  if (num_search_threads() < 1) {
    throw util::Exception("num_search_threads must be positive (%d)", num_search_threads());
  }
  if (params.enable_pondering && num_search_threads() == 1) {
    throw util::Exception("pondering mode does not work with only 1 search thread");
  }
  for (int i = 0; i < num_search_threads(); ++i) {
    search_threads_.push_back(new SearchThread(&shared_data_, nn_eval_service_, &params_, i));
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Manager<GameState, Tensorizor>::~Manager() {
  clear();
  if (nn_eval_service_) {
    nn_eval_service_->disconnect();
  }
  for (auto* thread : search_threads_) {
    delete thread;
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::start() {
  clear();
  shared_data_.root_softmax_temperature.reset();

  if (!connected_) {
    if (nn_eval_service_) {
      nn_eval_service_->connect();
    }
    connected_ = true;
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::clear() {
  stop_search_threads();
  if (!root_) return;

  assert(root_->parent()==nullptr);
  NodeReleaseService::release(root_);
  root_ = nullptr;
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::receive_state_change(
    core::seat_index_t seat, const GameState& state, core::action_index_t action)
{
  shared_data_.root_softmax_temperature.step();
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

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline const typename Manager<GameState, Tensorizor>::SearchResults* Manager<GameState, Tensorizor>::search(
    const Tensorizor& tensorizor, const GameState& game_state, const SearchParams& params)
{
  stop_search_threads();

  bool add_noise = !params.disable_exploration && params_.dirichlet_mult > 0;
  if (!root_ || add_noise) {
    if (root_) {
      NodeReleaseService::release(root_);
    }
    auto outcome = core::make_non_terminal_outcome<kNumPlayers>();
    root_ = new Node(tensorizor, game_state, outcome);  // TODO: use memory pool
  }

  start_search_threads(&params);
  wait_for_search_threads();

  const auto& evaluation_data = root_->evaluation_data();
  const auto& stable_data = root_->stable_data();

  auto evaluation = evaluation_data.ptr.load();
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

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::start_search_threads(const SearchParams* search_params) {
  assert(!shared_data_.search_active);
  shared_data_.search_active = true;
  num_active_search_threads_ = num_search_threads();

  for (auto* thread : search_threads_) {
    thread->launch(search_params, [&] { this->run_search(thread, search_params->tree_size_limit); });
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::wait_for_search_threads() {
  assert(shared_data_.search_active);

  for (auto* thread : search_threads_) {
    thread->join();
  }
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::stop_search_threads() {
  shared_data_.search_active = false;

  std::unique_lock<std::mutex> lock(search_mutex_);
  cv_search_.wait(lock, [&]{ return num_active_search_threads_ == 0; });
}

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::run_search(SearchThread* thread, int tree_size_limit) {
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

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Manager<GameState, Tensorizor>::get_cache_stats(
    int& hits, int& misses, int& size, float& hash_balance_factor) const
{
  nn_eval_service_->get_cache_stats(hits, misses, size, hash_balance_factor);
}

/*
 * The KataGo paper is a little vague in its description of the target pruning step, and examining the KataGo
 * source code was not very enlightening. The following is my best guess at what the target pruning step does.
 */
template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Manager<GameState, Tensorizor>::prune_counts(const SearchParams& search_params) {
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

template<core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Manager<GameState, Tensorizor>::init_profiling_dir(const std::string& profiling_dir) {
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

}  // namespace mcts
