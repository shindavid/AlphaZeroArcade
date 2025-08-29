#include "mcts/Algorithms.hpp"

#include "search/Constants.hpp"
#include "util/Asserts.hpp"
#include "util/BitSet.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"
#include "util/LoggingUtil.hpp"

#include <boost/algorithm/string/join.hpp>
#include <spdlog/spdlog.h>

#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace mcts {

template <typename Traits>
void Algorithms<Traits>::pure_backprop(SearchContext& context, const ValueArray& value) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, context.search_path_str(),
             fmt::streamed(value.transpose()));
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, true);
    stats.RN++;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::virtual_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  last_node->update_stats([&] {
    last_node->stats().VN++;  // thread-safe because executed under mutex
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().VN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::undo_virtual_backprop(SearchContext& context) {
  // NOTE: this is not an exact undo of virtual_backprop(), since the context.search_path is
  // modified in between the two calls.

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  RELEASE_ASSERT(!context.search_path.empty());

  for (int i = context.search_path.size() - 1; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E--;
      node->stats().VN--;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::standard_backprop(SearchContext& context, bool undo_virtual) {
  Node* last_node = context.search_path.back().node;
  auto value = GameResults::to_value_array(last_node->stable_data().VT);

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, context.search_path_str(),
             fmt::streamed(value.transpose()));
  }

  last_node->update_stats([&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    stats.init_q(value, false);
    stats.RN++;
    stats.VN -= undo_virtual;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E += !undo_virtual;
      auto& stats = node->stats();  // thread-safe because executed under mutex
      stats.RN++;
      stats.VN -= undo_virtual;
    });
  }
  validate_search_path(context);
}

template <typename Traits>
void Algorithms<Traits>::short_circuit_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    node->update_stats([&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  validate_search_path(context);
}

template <typename Traits>
bool Algorithms<Traits>::should_short_circuit(const Edge* edge, const Node* child) {
  int edge_count = edge->E;
  int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
  return edge_count < child_count;
}

template <typename Traits>
bool Algorithms<Traits>::more_search_iterations_needed(const GeneralContext& general_context,
                                                       const Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  const search::SearchParams& search_params = general_context.search_params;
  if (!search_params.ponder && root->trivial()) return false;
  return root->stats().total_count() <= search_params.tree_size_limit;
}

template <typename Traits>
void Algorithms<Traits>::init_root_info(GeneralContext& general_context,
                                        search::RootInitPurpose purpose) {
  const ManagerParams& manager_params = general_context.manager_params;
  const search::SearchParams& search_params = general_context.search_params;

  bool add_noise = false;
  switch (purpose) {
    case search::kForStandardSearch: {
      add_noise = search_params.full_search && manager_params.dirichlet_mult > 0;
      break;
    }
    case search::kToLoadRootActionValues: {
      add_noise = false;
      break;
    }
    default: {
      throw util::Exception("Unknown purpose {}", purpose);
    }
  }

  RootInfo& root_info = general_context.root_info;
  LookupTable* lookup_table = &general_context.lookup_table;

  root_info.add_noise = add_noise;
  if (root_info.node_index < 0 || add_noise) {
    const StateHistory& canonical_history = root_info.history_array[root_info.canonical_sym];
    root_info.node_index = lookup_table->alloc_node();
    Node* root = lookup_table->get_node(root_info.node_index);
    core::seat_index_t active_seat = Game::Rules::get_current_player(canonical_history.current());
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info.active_seat = active_seat;
    new (root) Node(lookup_table, canonical_history, active_seat);
  }

  Node* root2 = lookup_table->get_node(root_info.node_index);

  // thread-safe since single-threaded here
  if (root2->stats().RN == 0) {
    root2->stats().RN = 1;
  }

  if (search::kEnableSearchDebug && purpose == search::kForStandardSearch) {
    const auto& state = root_info.history_array[group::kIdentity].current();
    IO::print_state(std::cout, state);
  }
}

template <typename Traits>
int Algorithms<Traits>::get_best_child_index(const SearchContext& context) {
  const GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  Node* node = context.visit_node;
  bool is_root = (node == lookup_table.get_node(root_info.node_index));
  ActionSelector action_selector(manager_params, search_params, node, is_root);

  using PVec = LocalPolicyArray;

  const PVec& P = action_selector.P;
  const PVec& mE = action_selector.mE;
  PVec& PUCT = action_selector.PUCT;

  int argmax_index;

  if (search_params.tree_size_limit == 1) {
    // net-only, use P
    P.maxCoeff(&argmax_index);
  } else {
    bool force_playouts = manager_params.forced_playouts && is_root && search_params.full_search &&
                          manager_params.dirichlet_mult > 0;

    if (force_playouts) {
      PVec n_forced = (P * manager_params.k_forced * mE.sum()).sqrt();
      auto F1 = (mE < n_forced).template cast<float>();
      auto F2 = (mE > 0).template cast<float>();
      auto F = F1 * F2;
      PUCT = PUCT * (1 - F) + F * 1e+6;
    }

    PUCT.maxCoeff(&argmax_index);
  }

  print_action_selection_details(context, action_selector, argmax_index);
  return argmax_index;
}

template <typename Traits>
void Algorithms<Traits>::load_evaluations(SearchContext& context) {
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());
    node->load_eval(item.eval(), [&](LocalPolicyArray& P) { transform_policy(context, P); });
  }
}

template <typename Traits>
void Algorithms<Traits>::to_results(const GeneralContext& general_context, SearchResults& results) {
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  const Node* root = lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here

  core::action_mode_t mode = root->action_mode();
  group::element_t sym = root_info.canonical_sym;
  group::element_t inv_sym = SymmetryGroup::inverse(sym);

  results.valid_actions.reset();
  results.policy_prior.setZero();

  core::action_t actions[stable_data.num_valid_actions];

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data.valid_action_mask)) {
    Symmetries::apply(action, inv_sym, mode);
    results.valid_actions.set(action, true);
    actions[i] = action;

    auto* edge = root->get_edge(i);
    results.policy_prior(action) = edge->policy_prior_prob;

    i++;
  }

  load_action_symmetries(root, &actions[0], results);
  root->write_results(manager_params, inv_sym, results);
  results.policy_target = results.counts;
  results.provably_lost = stats.provably_losing[stable_data.active_seat];
  results.trivial = root->trivial();
  if (manager_params.forced_playouts && root_info.add_noise) {
    prune_policy_target(inv_sym, general_context, results);
  }

  Symmetries::apply(results.counts, inv_sym, mode);
  Symmetries::apply(results.policy_target, inv_sym, mode);
  Symmetries::apply(results.Q, inv_sym, mode);
  Symmetries::apply(results.Q_sq, inv_sym, mode);
  Symmetries::apply(results.action_values, inv_sym, mode);

  results.win_rates = stats.Q;
  results.value_prior = stable_data.VT;
  results.action_mode = mode;
}

template <typename Traits>
void Algorithms<Traits>::print_visit_info(const SearchContext& context) {
  if (search::kEnableSearchDebug) {
    const Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), context.search_path_str(),
             node->stable_data().active_seat);
  }
}

template <typename Traits>
void Algorithms<Traits>::transform_policy(SearchContext& context, LocalPolicyArray& P) {
  core::node_pool_index_t index = context.initialization_index;
  GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const ManagerParams& manager_params = general_context.manager_params;

  if (index == root_info.node_index) {
    if (search_params.full_search) {
      if (manager_params.dirichlet_mult) {
        add_dirichlet_noise(general_context, P);
      }
      P = P.pow(1.0 / general_context.aux_state.root_softmax_temperature.value());
      P /= P.sum();
    }
  }
}

template <typename Traits>
void Algorithms<Traits>::add_dirichlet_noise(GeneralContext& general_context, LocalPolicyArray& P) {
  const ManagerParams& manager_params = general_context.manager_params;
  auto& dirichlet_gen = general_context.aux_state.dirichlet_gen;
  auto& rng = general_context.aux_state.rng;

  int n = P.rows();
  double alpha = manager_params.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen.template generate<LocalPolicyArray>(rng, alpha, n);
  P = (1.0 - manager_params.dirichlet_mult) * P + manager_params.dirichlet_mult * noise;
}

template <typename Traits>
void Algorithms<Traits>::load_action_symmetries(const Node* root, core::action_t* actions,
                                                SearchResults& results) {
  const auto& stable_data = root->stable_data();

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_actions);

  for (int e = 0; e < stable_data.num_valid_actions; ++e) {
    Edge* edge = root->get_edge(e);
    if (edge->child_index < 0) continue;
    items.emplace_back(edge->child_index, actions[e]);
  }

  results.action_symmetry_table.load(items);
}

template <typename Traits>
void Algorithms<Traits>::prune_policy_target(group::element_t inv_sym,
                                             const GeneralContext& general_context,
                                             SearchResults& results) {
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  if (manager_params.no_model) return;

  const Node* root = lookup_table.get_node(root_info.node_index);
  ActionSelector action_selector(manager_params, search_params, root, true);

  const auto& P = action_selector.P;
  const auto& E = action_selector.E;
  const auto& PW = action_selector.PW;
  const auto& PL = action_selector.PL;
  const auto& mE = action_selector.mE;
  const auto& Q = action_selector.Q;
  const auto& PUCT = action_selector.PUCT;

  auto mE_sum = mE.sum();
  auto n_forced = (P * manager_params.k_forced * mE_sum).sqrt();

  int mE_max_index;
  auto mE_max = mE.maxCoeff(&mE_max_index);

  auto PUCT_max = PUCT(mE_max_index);
  auto sqrt_mE = sqrt(mE_sum + ActionSelector::eps);

  LocalPolicyArray mE_floor = manager_params.cPUCT * P * sqrt_mE / (PUCT_max - 2 * Q) - 1;

  int n_actions = root->stable_data().num_valid_actions;
  for (int i = 0; i < n_actions; ++i) {
    const Edge* edge = root->get_edge(i);
    if (mE(i) == 0) {
      results.policy_target(edge->action) = 0;
      continue;
    }
    if (mE(i) == mE_max) continue;
    if (!isfinite(mE_floor(i))) continue;
    if (mE_floor(i) >= mE(i)) continue;
    auto n = std::max(mE_floor(i), mE(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }
    results.policy_target(edge->action) = n;
  }

  if (eigen_util::sum(results.policy_target) <= 0) {
    // can happen in certain edge cases
    results.policy_target = results.counts;
  }

  if (search::kEnableSearchDebug) {
    LocalPolicyArray actions(n_actions);
    LocalPolicyArray pruned(n_actions);

    core::action_mode_t mode = root->action_mode();
    for (int i = 0; i < n_actions; ++i) {
      core::action_t raw_action = root->get_edge(i)->action;
      core::action_t action = raw_action;
      Symmetries::apply(action, inv_sym, mode);
      actions(i) = action;
      pruned(i) = results.policy_target(raw_action);
    }

    LocalPolicyArray target = pruned / pruned.sum();

    static std::vector<std::string> columns = {"action", "P",  "Q",  "PUCT",   "E",
                                               "PW",     "PL", "mE", "pruned", "target"};
    auto data = eigen_util::sort_rows(
      eigen_util::concatenate_columns(actions, P, Q, PUCT, E, PW, PL, mE, pruned, target));

    eigen_util::PrintArrayFormatMap fmt_map{
      {"action", [&](float x) { return IO::action_to_str(x, mode); }},
    };

    std::cout << std::endl << "Policy target pruning:" << std::endl;
    eigen_util::print_array(std::cout, data, columns, &fmt_map);
  }
}

template <typename Traits>
void Algorithms<Traits>::validate_search_path(const SearchContext& context) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  int N = context.search_path.size();
  for (int i = N - 1; i >= 0; --i) {
    context.search_path[i].node->validate_state();
  }
}

template <typename Traits>
void Algorithms<Traits>::print_action_selection_details(const SearchContext& context,
                                                        const ActionSelector& selector,
                                                        int argmax_index) {
  Node* node = context.visit_node;
  if (search::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << std::format("{:>{}}", "", context.log_prefix_n());

    core::seat_index_t seat = node->stable_data().active_seat;

    int n_actions = node->stable_data().num_valid_actions;

    ValueArray players;
    ValueArray nQ = node->stats().Q;
    ValueArray CP;
    for (int p = 0; p < kNumPlayers; ++p) {
      players(p) = p;
      CP(p) = p == seat;
    }

    static std::vector<std::string> player_columns = {"Seat", "Q", "CurP"};
    auto player_data = eigen_util::concatenate_columns(players, nQ, CP);

    eigen_util::PrintArrayFormatMap fmt_map1{
      {"Seat", [&](float x) { return std::to_string(int(x)); }},
      {"CurP", [&](float x) { return std::string(x == seat ? "*" : ""); }},
    };

    std::stringstream ss1;
    eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

    std::string line_break =
      std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context.log_prefix_n());

    for (const std::string& line : util::splitlines(ss1.str())) {
      ss << line << line_break;
    }

    const LocalPolicyArray& P = selector.P;
    const LocalPolicyArray& Q = selector.Q;
    const LocalPolicyArray& FPU = selector.FPU;
    const LocalPolicyArray& PW = selector.PW;
    const LocalPolicyArray& PL = selector.PL;
    const LocalPolicyArray& E = selector.E;
    const LocalPolicyArray& mE = selector.mE;
    const LocalPolicyArray& RN = selector.RN;
    const LocalPolicyArray& VN = selector.VN;
    const LocalPolicyArray& PUCT = selector.PUCT;

    LocalPolicyArray actions(n_actions);
    LocalPolicyArray child_addr(n_actions);
    LocalPolicyArray argmax(n_actions);
    child_addr.setConstant(-1);
    argmax.setZero();
    argmax(argmax_index) = 1;

    group::element_t inv_sym = SymmetryGroup::inverse(context.leaf_canonical_sym);
    for (int e = 0; e < n_actions; ++e) {
      auto edge = node->get_edge(e);
      core::action_t action = edge->action;
      Symmetries::apply(action, inv_sym, node->action_mode());
      actions(e) = action;
      child_addr(e) = edge->child_index;
    }

    static std::vector<std::string> action_columns = {
      "action", "P", "Q", "FPU", "PW", "PL", "E", "mE", "RN", "VN", "&ch", "PUCT", "argmax"};
    auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
      actions, P, Q, FPU, PW, PL, E, mE, RN, VN, child_addr, PUCT, argmax));

    eigen_util::PrintArrayFormatMap fmt_map2{
      {"action", [&](float x) { return IO::action_to_str(x, node->action_mode()); }},
      {"&ch", [](float x) { return x < 0 ? std::string() : std::to_string((int)x); }},
      {"argmax", [](float x) { return std::string(x == 0 ? "" : "*"); }},
    };

    std::stringstream ss2;
    eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

    for (const std::string& line : util::splitlines(ss2.str())) {
      ss << line << line_break;
    }

    LOG_INFO(ss.str());
  }
}

}  // namespace mcts
