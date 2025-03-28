#include <mcts/Manager.hpp>

#include <mcts/ActionSelector.hpp>
#include <mcts/TypeDefs.hpp>
#include <mcts/UniformNNEvaluationService.hpp>
#include <util/Asserts.hpp>
#include <util/Exception.hpp>

#include <boost/filesystem.hpp>

#include <memory>

namespace mcts {

template <core::concepts::Game Game>
int Manager<Game>::next_instance_id_ = 0;

template <core::concepts::Game Game>
Manager<Game>::Manager(bool dummy, mutex_cv_vec_sptr_t mutex_cv_pool,
                       const ManagerParams& params, NNEvaluationServiceBase* service)
    : params_(params),
      pondering_search_params_(
          SearchParams::make_pondering_params(params.pondering_tree_size_limit)),
      shared_data_(mutex_cv_pool, params, next_instance_id_++) {
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

  if (service) {
    nn_eval_service_ = service;
  } else if (!params.no_model) {
    nn_eval_service_ = NNEvaluationService::create(params);
  } else if (params.model_filename.empty()) {
    nn_eval_service_ = new UniformNNEvaluationService<Game>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }

  if (num_search_threads() < 1) {
    throw util::CleanException("num_search_threads must be positive (%d)", num_search_threads());
  }
  if (params.enable_pondering && num_search_threads() == 1) {
    throw util::CleanException("pondering mode does not work with only 1 search thread");
  }
  for (int i = 0; i < num_search_threads(); ++i) {
    auto thread = new SearchThread(&shared_data_, nn_eval_service_, &params_, i);
    if (mcts::kEnableProfiling) {
      thread->set_profiling_dir(params.profiling_dir());
    }
    search_threads_.push_back(thread);
  }
}

template <core::concepts::Game Game>
Manager<Game>::Manager(const ManagerParams& params, NNEvaluationServiceBase* service)
    : Manager(true, std::make_shared<mutex_cv_vec_t>(1), params, service) {}

template <core::concepts::Game Game>
Manager<Game>::Manager(mutex_cv_vec_sptr_t& mutex_cv_pool, const ManagerParams& params,
                       NNEvaluationServiceBase* service)
    : Manager(true, mutex_cv_pool, params, service) {}

template <core::concepts::Game Game>
inline Manager<Game>::~Manager() {
  announce_shutdown();
  clear();

  nn_eval_service_->disconnect();
  for (auto* thread : search_threads_) {
    delete thread;
  }
}

template <core::concepts::Game Game>
inline void Manager<Game>::start_threads() {
  for (SearchThread* thread : search_threads_) {
    thread->start();
  }
}

template <core::concepts::Game Game>
inline void Manager<Game>::start() {
  clear();

  if (!connected_) {
    nn_eval_service_->connect();
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
                                                const State&,
                                                core::action_t action) {
  group::element_t sym = shared_data_.root_info.canonical_sym;

  core::action_mode_t mode = shared_data_.get_current_action_mode();
  shared_data_.update_state(action);
  shared_data_.root_softmax_temperature.step();
  stop_search_threads();  // stop pondering
  node_pool_index_t root_index = shared_data_.root_info.node_index;
  if (root_index < 0) return;

  core::action_t transformed_action = action;
  Game::Symmetries::apply(transformed_action, sym, mode);

  Node* root = shared_data_.lookup_table.get_node(root_index);
  root_index = root->lookup_child_by_action(transformed_action);  // tree reuse
  if (root_index < 0) {
    shared_data_.root_info.node_index = -1;
    return;
  }

  shared_data_.root_info.node_index = root_index;

  if (params_.enable_pondering) {
    start_search_threads(pondering_search_params_);  // resume pondering
  }
}

template <core::concepts::Game Game>
inline const typename Manager<Game>::SearchResults*
Manager<Game>::search(const SearchParams& params) {
  stop_search_threads();  // stop pondering

  bool add_noise = params.full_search && params_.dirichlet_mult > 0;
  shared_data_.init_root_info(add_noise);

  auto& root_info = shared_data_.root_info;
  if (mcts::kEnableSearchDebug) {
    Game::IO::print_state(std::cout, root_info.history_array[group::kIdentity].current());
  }

  start_search_threads(params);
  wait_for_search_threads();

  shared_data_.lookup_table.defragment(root_info.node_index);
  Node* root = shared_data_.lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = root_info.canonical_sym;
  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);

  results_.valid_actions.reset();
  results_.policy_prior.setZero();

  core::action_t actions[stable_data.num_valid_actions];

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data.valid_action_mask)) {
    Game::Symmetries::apply(action, inv_sym, mode);
    results_.valid_actions.set(action, true);
    actions[i] = action;

    auto* edge = root->get_edge(i);
    results_.policy_prior(action) = edge->base_prob;

    i++;
  }

  load_action_symmetries(root, &actions[0]);
  root->write_results(params_, inv_sym, results_);
  results_.policy_target = results_.counts;
  results_.provably_lost = stats.provably_losing[stable_data.active_seat];
  results_.trivial = root->trivial();
  if (params_.forced_playouts && add_noise) {
    prune_policy_target(params, inv_sym);
  }

  Game::Symmetries::apply(results_.counts, inv_sym, mode);
  Game::Symmetries::apply(results_.policy_target, inv_sym, mode);
  Game::Symmetries::apply(results_.Q, inv_sym, mode);
  Game::Symmetries::apply(results_.Q_sq, inv_sym, mode);
  Game::Symmetries::apply(results_.action_values, inv_sym, mode);

  results_.win_rates = stats.Q;
  results_.value_prior = stable_data.VT;
  results_.action_mode = mode;

  return &results_;
}

/*
 * Here, we do a skimmed-down version of Manager::search()
 */
template <core::concepts::Game Game>
inline void Manager<Game>::load_root_action_values(ActionValueTensor& action_values) {
  action_values.setZero();
  stop_search_threads();  // stop pondering

  shared_data_.init_root_info(false);
  auto& root_info = shared_data_.root_info;

  // We do a dummy search with 0 iterations, just to get SearchThread to call init_root_node(),
  // which will expand all the root's children.
  constexpr int tree_size_limit = 0;
  constexpr bool full_search = true;
  constexpr bool ponder = false;
  SearchParams params{tree_size_limit, full_search, ponder};

  start_search_threads(params);
  wait_for_search_threads();

  Node* root = shared_data_.lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = root_info.canonical_sym;

  util::release_assert(Game::Rules::is_chance_mode(mode));

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data.valid_action_mask)) {
    auto* edge = root->get_edge(i);
    core::action_t transformed_action = action;
    Game::Symmetries::apply(transformed_action, sym, mode);
    node_pool_index_t child_node_index = root->lookup_child_by_action(transformed_action);
    if (child_node_index < 0) {
      action_values(action) = edge->child_V_estimate;
    } else {
      Node* child = shared_data_.lookup_table.get_node(child_node_index);
      action_values(action) = child->stable_data().VT(stable_data.active_seat);
    }
    i++;
  }

  if (params_.enable_pondering) {
    start_search_threads(pondering_search_params_);  // resume pondering
  }
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
inline void Manager<Game>::announce_shutdown() {
  std::unique_lock<std::mutex> lock(shared_data_.search_mutex);
  shared_data_.shutting_down = true;
  shared_data_.cv_search_on.notify_all();
}

template <core::concepts::Game Game>
inline void Manager<Game>::load_action_symmetries(Node* root, core::action_t* actions) {
  const auto& stable_data = root->stable_data();

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_actions);

  for (int e = 0; e < stable_data.num_valid_actions; ++e) {
    Edge* edge = root->get_edge(e);
    if (edge->child_index < 0) continue;
    items.emplace_back(edge->child_index, actions[e]);
  }

  results_.action_symmetry_table.load(items);
}

template <core::concepts::Game Game>
void Manager<Game>::prune_policy_target(const SearchParams& search_params,
                                        group::element_t inv_sym) {
  using ActionSelector = mcts::ActionSelector<Game>;

  if (params_.no_model) return;

  Node* root = shared_data_.get_root_node();
  ActionSelector action_selector(params_, search_params, root, true);

  const auto& P = action_selector.P;
  const auto& E = action_selector.E;
  const auto& PW = action_selector.PW;
  const auto& PL = action_selector.PL;
  const auto& mE = action_selector.mE;
  const auto& Q = action_selector.Q;
  const auto& PUCT = action_selector.PUCT;

  auto mE_sum = mE.sum();
  auto n_forced = (P * params_.k_forced * mE_sum).sqrt();

  int mE_max_index;
  auto mE_max = mE.maxCoeff(&mE_max_index);

  auto PUCT_max = PUCT(mE_max_index);
  auto sqrt_mE = sqrt(mE_sum + ActionSelector::eps);

  LocalPolicyArray mE_floor = params_.cPUCT * P * sqrt_mE / (PUCT_max - 2 * Q) - 1;

  int n_actions = root->stable_data().num_valid_actions;
  for (int i = 0; i < n_actions; ++i) {
    Edge* edge = root->get_edge(i);
    if (mE(i) == 0) {
      results_.policy_target(edge->action) = 0;
      continue;
    }
    if (mE(i) == mE_max) continue;
    if (!isfinite(mE_floor(i))) continue;
    if (mE_floor(i) >= mE(i)) continue;
    auto n = std::max(mE_floor(i), mE(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }
    results_.policy_target(edge->action) = n;
  }

  if (eigen_util::sum(results_.policy_target) <= 0) {
    // can happen in certain edge cases
    results_.policy_target = results_.counts;
  }

  if (mcts::kEnableSearchDebug) {
    LocalPolicyArray actions(n_actions);
    LocalPolicyArray pruned(n_actions);

    core::action_mode_t mode = root->action_mode();
    for (int i = 0; i < n_actions; ++i) {
      core::action_t raw_action = root->get_edge(i)->action;
      core::action_t action = raw_action;
      Game::Symmetries::apply(action, inv_sym, mode);
      actions(i) = action;
      pruned(i) = results_.policy_target(raw_action);
    }

    LocalPolicyArray target = pruned / pruned.sum();

    static std::vector<std::string> columns = {"action", "P",  "Q",  "PUCT",  "E",
                                               "PW",     "PL", "mE", "pruned", "target"};
    auto data = eigen_util::sort_rows(
        eigen_util::concatenate_columns(actions, P, Q, PUCT, E, PW, PL, mE, pruned, target));

    eigen_util::PrintArrayFormatMap fmt_map {
      {"action", [&](float x) { return Game::IO::action_to_str(x, mode); }},
    };

    std::cout << std::endl << "Policy target pruning:" << std::endl;
    eigen_util::print_array(std::cout, data, columns, &fmt_map);
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

template <core::concepts::Game Game>
void Manager<Game>::set_post_visit_func(std::function<void()> func) {
  for (SearchThread* thread : search_threads_) {
    thread->set_post_visit_func(func);
  }
}

}  // namespace mcts
