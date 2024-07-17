#include <mcts/Manager.hpp>

#include <boost/filesystem.hpp>

#include <core/GameVars.hpp>
#include <mcts/PUCTStats.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/Asserts.hpp>
#include <util/Exception.hpp>

namespace mcts {

template <core::concepts::Game Game>
int Manager<Game>::next_instance_id_ = 0;

template <core::concepts::Game Game>
inline Manager<Game>::Manager(const ManagerParams& params)
    : params_(params),
      pondering_search_params_(
          SearchParams::make_pondering_params(params.pondering_tree_size_limit)),
      shared_data_(num_search_threads() > 1) {
  shared_data_.manager_id = next_instance_id_++;
  new (&shared_data_.root_softmax_temperature) math::ExponentialDecay(math::ExponentialDecay::parse(
      params.root_softmax_temperature_str, core::GameVars<Game>::get_bindings()));
  namespace bf = boost::filesystem;

  if (mcts::kEnableProfiling) {
    auto profiling_dir = params_.profiling_dir();
    if (profiling_dir.empty()) {
      throw util::CleanException(
          "Required: --mcts-profiling-dir. Alternatively, add entry for 'mcts_profiling_dir' in "
          "config.txt");
    }
    init_profiling_dir(profiling_dir.string());
  }

  if (!params.no_model) {
    nn_eval_service_ = NNEvaluationService::create(params);
    if (mcts::kEnableProfiling) {
      nn_eval_service_->set_profiling_dir(params.profiling_dir());
    }
  } else if (!params.model_filename.empty()) {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }

  if (num_search_threads() < 1) {
    throw util::CleanException("num_search_threads must be positive (%d)", num_search_threads());
  }
  if (params.enable_pondering && num_search_threads() == 1) {
    throw util::CleanException("pondering mode does not work with only 1 search thread");
  }
  shared_data_.active_search_threads.resize(num_search_threads());
  for (int i = 0; i < num_search_threads(); ++i) {
    auto thread = new SearchThread(&shared_data_, nn_eval_service_, &params_, i);
    if (mcts::kEnableProfiling) {
      thread->set_profiling_dir(params.profiling_dir());
    }
    search_threads_.push_back(thread);
  }
}

template <core::concepts::Game Game>
inline Manager<Game>::~Manager() {
  announce_shutdown();
  clear();
  if (nn_eval_service_) {
    nn_eval_service_->disconnect();
  }
  for (auto* thread : search_threads_) {
    delete thread;
  }
}

template <core::concepts::Game Game>
inline void Manager<Game>::announce_shutdown() {
  std::unique_lock<std::mutex> lock(shared_data_.search_mutex);
  shared_data_.shutting_down = true;
  shared_data_.cv_search_on.notify_all();
}

template <core::concepts::Game Game>
inline void Manager<Game>::start() {
  clear();

  if (!connected_) {
    if (nn_eval_service_) {
      nn_eval_service_->connect();
    }
    connected_ = true;
  }
}

template <core::concepts::Game Game>
inline void Manager<Game>::clear() {
  stop_search_threads();
  shared_data_.clear();
}

template <core::concepts::Game Game>
inline void Manager<Game>::receive_state_change(core::seat_index_t seat,
                                                const FullState& state,
                                                core::action_t action) {
  using node_pool_index_t = Node::node_pool_index_t;

  group::element_t sym = shared_data_.root_info.canonical_sym;

  shared_data_.update_state(action);
  shared_data_.root_softmax_temperature.step();
  stop_search_threads();
  node_pool_index_t root_index = shared_data_.root_info.node_index;
  if (root_index < 0) return;

  core::action_t transformed_action = action;
  Game::Symmetries::apply(transformed_action, sym);

  Node* root = shared_data_.lookup_table.get_node(root_index);
  root_index = root->lookup_child_by_action(transformed_action);
  if (root_index < 0) {
    shared_data_.root_info.node_index = -1;
    return;
  }

  shared_data_.root_info.node_index = root_index;

  if (params_.enable_pondering) {
    start_search_threads(pondering_search_params_);
  }
}

template <core::concepts::Game Game>
inline const typename Manager<Game>::SearchResults*
Manager<Game>::search(const FullState& game_state, const SearchParams& params) {
  using ActionOutcome = Game::Types::ActionOutcome;

  stop_search_threads();

  auto& root_info = shared_data_.root_info;
  bool add_noise = !params.disable_exploration && params_.dirichlet_mult > 0;
  if (root_info.node_index < 0 || add_noise) {
    const FullState& canonical_state = root_info.state[root_info.canonical_sym];
    ActionOutcome outcome;
    root_info.node_index = shared_data_.lookup_table.alloc_node();
    Node* root = shared_data_.lookup_table.get_node(root_info.node_index);
    new (root) Node(&shared_data_.lookup_table, canonical_state, outcome);
  }

  if (mcts::kEnableDebug) {
    Game::IO::print_state(std::cout, root_info.state[group::kIdentity]);
  }

  start_search_threads(params);
  wait_for_search_threads();

  shared_data_.lookup_table.defragment(root_info.node_index);
  Node* root = shared_data_.lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();

  group::element_t sym = root_info.canonical_sym;
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);

  results_.valid_actions.set(false);
  results_.policy_prior.setZero();

  core::action_t actions[stable_data.num_valid_actions];

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data.valid_action_mask)) {
    Game::Symmetries::apply(action, inv_sym);
    results_.valid_actions.set(action, true);
    actions[i] = action;

    auto* edge = root->get_edge(i);
    results_.policy_prior(action) = edge->raw_policy_prior;

    i++;
  }

  std::array<core::action_t, Game::Constants::kNumActions> action_collapse_lookup;
  action_collapse_lookup.fill(-1);
  for (int e = 0; e < stable_data.num_valid_actions; ++e) {
    action_collapse_lookup[actions[e]] = actions[root->get_edge(e)->representative_edge_index];
  }

  results_.action_collapse_table.load(action_collapse_lookup);
  results_.counts = root->get_counts(params_);
  results_.policy_target = results_.counts;
  results_.provably_lost = stats.provably_losing[stable_data.current_player];
  results_.num_representative_actions = root->num_representative_actions();
  if (params_.forced_playouts && add_noise) {
    prune_policy_target(params);
  }

  Game::Symmetries::apply(results_.counts, inv_sym);
  Game::Symmetries::apply(results_.policy_target, inv_sym);

  results_.win_rates = stats.RQ;
  results_.value_prior = stable_data.V;

  return &results_;
}

template <core::concepts::Game Game>
inline void Manager<Game>::start_search_threads(const SearchParams& search_params) {
  std::unique_lock<std::mutex> lock(shared_data_.search_mutex);

  shared_data_.search_params = search_params;
  shared_data_.active_search_threads.set();
  shared_data_.cv_search_on.notify_all();
}

template <core::concepts::Game Game>
inline void Manager<Game>::wait_for_search_threads() {
  std::unique_lock<std::mutex> lock(shared_data_.search_mutex);
  shared_data_.cv_search_off.wait(lock, [&] { return shared_data_.active_search_threads.none(); });
}

template <core::concepts::Game Game>
inline void Manager<Game>::stop_search_threads() {
  std::unique_lock<std::mutex> lock(shared_data_.search_mutex);
  shared_data_.search_params.tree_size_limit = 0;
  shared_data_.cv_search_off.wait(lock, [&] { return shared_data_.active_search_threads.none(); });
}

template <core::concepts::Game Game>
void Manager<Game>::prune_policy_target(const SearchParams& search_params) {
  using PUCTStats = mcts::PUCTStats<Game>;
  using edge_t = Node::edge_t;

  if (params_.no_model) return;

  Node* root = shared_data_.get_root_node();
  PUCTStats stats(params_, search_params, root, true);

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

  for (int i = 0; i < root->stable_data().num_valid_actions; ++i) {
    edge_t* edge = root->get_edge(i);
    if (N(i) == N_max) continue;
    if (!isfinite(N_floor(i))) continue;
    auto n = std::max(N_floor(i), N(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }

    results_.policy_target(edge->action) = n;
  }

  const auto& policy_target_array = eigen_util::reinterpret_as_array(results_.policy_target);
  if (policy_target_array.sum() <= 0) {
    // can happen in certain edge cases
    results_.policy_target = results_.counts;
    return;
  }
}

template <core::concepts::Game Game>
void Manager<Game>::init_profiling_dir(const std::string& profiling_dir) {
  static std::string pdir;
  if (!pdir.empty()) {
    if (pdir == profiling_dir) return;
    throw util::CleanException("Two different mcts profiling dirs used: %s and %s", pdir.c_str(),
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
