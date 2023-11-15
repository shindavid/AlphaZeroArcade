#include <mcts/Manager.hpp>

#include <boost/filesystem.hpp>

#include <mcts/TypeDefs.hpp>
#include <util/Asserts.hpp>
#include <util/Exception.hpp>

namespace mcts {

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
int Manager<GameState, Tensorizor>::next_instance_id_ = 0;

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Manager<GameState, Tensorizor>::Manager(const ManagerParams& params)
    : params_(params),
      pondering_search_params_(
          SearchParams::make_pondering_params(params.pondering_tree_size_limit)) {
  tree_data_.set_manager_id(next_instance_id_++);
  tree_data_.set_root_softmax_temperature(math::ExponentialDecay(math::ExponentialDecay::parse(
      params.root_softmax_temperature_str, GameStateTypes::get_var_bindings())));
  namespace bf = boost::filesystem;

  if (mcts::kEnableProfiling) {
    auto profiling_dir = params_.profiling_dir();
    if (profiling_dir.empty()) {
      throw util::Exception(
          "Required: --mcts-profiling-dir. Alternatively, add entry for 'mcts_profiling_dir' in "
          "config.txt");
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
  search_thread_ = new SearchThread(&tree_data_, nn_eval_service_, &params_);
  prefetch_manager_ = PrefetchThreadManager::get(params);
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline Manager<GameState, Tensorizor>::~Manager() {
  clear();
  tree_data_.shutdown();
  prefetch_manager_->shutdown();
  delete search_thread_;
  if (nn_eval_service_) {
    nn_eval_service_->disconnect();
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::start() {
  clear();
  tree_data_.reset_move_number();
  tree_data_.reset_root_softmax_temperature();
  tree_data_.node_cache().clear();

  if (!connected_) {
    if (nn_eval_service_) {
      nn_eval_service_->connect();
    }
    connected_ = true;
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::clear() {
  tree_data_.wait_for_search_completion();
  tree_data_.set_root_node(nullptr);
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline void Manager<GameState, Tensorizor>::receive_state_change(core::seat_index_t seat,
                                                                 const GameState& state,
                                                                 const Action& action) {
  tree_data_.increment_move_number();
  tree_data_.node_cache().clear_before(tree_data_.move_number());
  tree_data_.step_root_softmax_temperature();
  tree_data_.wait_for_search_completion();
  auto root_node = tree_data_.root_node();
  if (!root_node.get()) return;

  auto new_root = root_node->lookup_child_by_action(action);
  if (!new_root.get()) {
    tree_data_.set_root_node(nullptr);
    return;
  }

  tree_data_.set_root_node(new_root);

  if (params_.enable_pondering) {
    tree_data_.activate_search();
    prefetch_manager_->add_work(&tree_data_, nn_eval_service_, &pondering_search_params_,
                                &params_);
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
inline const typename Manager<GameState, Tensorizor>::SearchResults*
Manager<GameState, Tensorizor>::search(const Tensorizor& tensorizor, const GameState& game_state,
                                       const SearchParams& params) {
  if (params_.enable_pondering) {
    tree_data_.deactivate_search();
    tree_data_.wait_for_search_completion();
  }

  bool add_noise = !params.disable_exploration && params_.dirichlet_mult > 0;
  if (!tree_data_.root_node() || add_noise) {
    auto outcome = GameStateTypes::make_non_terminal_outcome();
    tree_data_.set_root_node(std::make_shared<Node>(tensorizor, game_state, outcome,
                                                    &params_));  // TODO: use memory pool
  }

  if (mcts::kEnableDebug) {
    const GameState& state = tree_data_.root_node()->stable_data().state;
    state.dump();
  }

  search_thread_->set_search_params(&params);
  tree_data_.activate_search();
  prefetch_manager_->add_work(&tree_data_, nn_eval_service_, &params, &params_);
  tree_data_.wait_for_search_completion();

  const auto& root = tree_data_.root_node();
  const auto& evaluation_data = root->evaluation_data();
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats(kSearchMode);

  auto evaluation = evaluation_data.ptr.load();
  results_.valid_actions = stable_data.valid_action_mask;
  results_.counts = root->get_counts(params_);
  results_.provably_lost = stats.provably_losing[stable_data.current_player];
  if (params_.forced_playouts && add_noise) {
    prune_counts(params);
  }
  results_.policy_prior = evaluation_data.local_policy_prob_distr;
  results_.win_rates = stats.real_avg;
  results_.value_prior = evaluation->value_prob_distr();
  return &results_;
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Manager<GameState, Tensorizor>::get_cache_stats(int& hits, int& misses, int& size,
                                                     float& hash_balance_factor) const {
  nn_eval_service_->get_cache_stats(hits, misses, size, hash_balance_factor);
}

/*
 * The KataGo paper is a little vague in its description of the target pruning step, and examining
 * the KataGo source code was not very enlightening. The following is my best guess at what the
 * target pruning step does.
 */
template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Manager<GameState, Tensorizor>::prune_counts(const SearchParams& search_params) {
  if (params_.model_filename.empty()) return;

  PUCTStats stats(params_, search_params, kSearchMode, tree_data_.root_node().get(), true);

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

  for (auto& it : tree_data_.root_node()->children_data()) {
    core::action_index_t i = it.action_index();
    if (N(i) == N_max) continue;
    if (!isfinite(N_floor(i))) continue;
    auto n = std::max(N_floor(i), N(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }

    results_.counts(it.action()) = n;
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
    std::cout << "orig_counts: " << eigen_util::reinterpret_as_array(orig_counts).transpose()
              << std::endl;
    std::cout << "results_.counts: " << counts_array.transpose() << std::endl;
    throw util::Exception("prune_counts: counts problem");
  }
}

template <core::GameStateConcept GameState, core::TensorizorConcept<GameState> Tensorizor>
void Manager<GameState, Tensorizor>::init_profiling_dir(const std::string& profiling_dir) {
  static std::string pdir;
  if (!pdir.empty()) {
    if (pdir == profiling_dir) return;
    throw util::Exception("Two different mcts profiling dirs used: %s and %s", pdir.c_str(),
                          profiling_dir.c_str());
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
