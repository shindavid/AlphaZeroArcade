#include "alphazero/Algorithms.hpp"

#include "search/Constants.hpp"
#include "util/Asserts.hpp"
#include "util/CppUtil.hpp"
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

namespace alpha0 {

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::pure_backprop(SearchContext& context,
                                                    const ValueArray& value) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, context.search_path_str(),
             fmt::streamed(value.transpose()));
  }

  LookupTable& lookup_table = context.general_context->lookup_table;
  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  Derived::backprop_helper(last_node, lookup_table, [&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    Derived::init_q(stats, value, true);
    stats.RN++;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    Derived::backprop_helper(node, lookup_table, [&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  Derived::validate_search_path(context);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::virtual_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  LookupTable& lookup_table = context.general_context->lookup_table;
  RELEASE_ASSERT(!context.search_path.empty());
  Node* last_node = context.search_path.back().node;

  Derived::backprop_helper(last_node, lookup_table, [&] {
    last_node->stats().VN++;  // thread-safe because executed under mutex
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    Derived::backprop_helper(node, lookup_table, [&] {
      edge->E++;
      node->stats().VN++;  // thread-safe because executed under mutex
    });
  }
  Derived::validate_search_path(context);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::undo_virtual_backprop(SearchContext& context) {
  // NOTE: this is not an exact undo of virtual_backprop(), since the context.search_path is
  // modified in between the two calls.

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  LookupTable& lookup_table = context.general_context->lookup_table;
  RELEASE_ASSERT(!context.search_path.empty());

  for (int i = context.search_path.size() - 1; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    Derived::backprop_helper(node, lookup_table, [&] {
      edge->E--;
      node->stats().VN--;  // thread-safe because executed under mutex
    });
  }
  Derived::validate_search_path(context);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::standard_backprop(SearchContext& context, bool undo_virtual) {
  Node* last_node = context.search_path.back().node;
  auto value = GameResults::to_value_array(last_node->stable_data().R);

  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {} {}", "", context.log_prefix_n(), __func__, context.search_path_str(),
             fmt::streamed(value.transpose()));
  }

  LookupTable& lookup_table = context.general_context->lookup_table;

  Derived::backprop_helper(last_node, lookup_table, [&] {
    auto& stats = last_node->stats();  // thread-safe because executed under mutex
    Derived::init_q(stats, value, false);
    stats.RN++;
    stats.VN -= undo_virtual;
  });

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    Derived::backprop_helper(node, lookup_table, [&] {
      edge->E += !undo_virtual;
      auto& stats = node->stats();  // thread-safe because executed under mutex
      stats.RN++;
      stats.VN -= undo_virtual;
    });
  }
  Derived::validate_search_path(context);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::short_circuit_backprop(SearchContext& context) {
  LOG_TRACE("{:>{}}{}()", "", context.log_prefix_n(), __func__);
  if (search::kEnableSearchDebug) {
    LOG_INFO("{:>{}}{} {}", "", context.log_prefix_n(), __func__, context.search_path_str());
  }

  LookupTable& lookup_table = context.general_context->lookup_table;

  for (int i = context.search_path.size() - 2; i >= 0; --i) {
    Edge* edge = context.search_path[i].edge;
    Node* node = context.search_path[i].node;

    // NOTE: always update the edge first, then the parent node
    Derived::backprop_helper(node, lookup_table, [&] {
      edge->E++;
      node->stats().RN++;  // thread-safe because executed under mutex
    });
  }
  Derived::validate_search_path(context);
}

template <search::concepts::Traits Traits, typename Derived>
bool AlgorithmsBase<Traits, Derived>::should_short_circuit(const Edge* edge, const Node* child) {
  int edge_count = edge->E;
  int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
  return edge_count < child_count;
}

template <search::concepts::Traits Traits, typename Derived>
bool AlgorithmsBase<Traits, Derived>::more_search_iterations_needed(
  const GeneralContext& general_context, const Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  const search::SearchParams& search_params = general_context.search_params;
  if (!search_params.ponder && root->trivial()) return false;
  return root->stats().total_count() <= search_params.tree_size_limit;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_root_info(GeneralContext& general_context,
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
  LookupTable& lookup_table = general_context.lookup_table;

  root_info.add_noise = add_noise;
  if (root_info.node_index < 0 || add_noise) {
    root_info.node_index = lookup_table.alloc_node();
    Node* root = lookup_table.get_node(root_info.node_index);

    StateHistory history = root_info.history;  // copy
    for (auto& state : history) {
      Symmetries::apply(state, root_info.canonical_sym);
    }
    State& cur_state = history.current();
    core::seat_index_t active_seat = Game::Rules::get_current_player(cur_state);
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info.active_seat = active_seat;
    new (root) Node(lookup_table.get_random_mutex(), cur_state, active_seat);
  }

  Node* root2 = lookup_table.get_node(root_info.node_index);

  // thread-safe since single-threaded here
  if (root2->stats().RN == 0) {
    root2->stats().RN = 1;
  }

  if (search::kEnableSearchDebug && purpose == search::kForStandardSearch) {
    const auto& state = root_info.history.current();
    IO::print_state(std::cout, state);
  }
}

template <search::concepts::Traits Traits, typename Derived>
int AlgorithmsBase<Traits, Derived>::get_best_child_index(const SearchContext& context) {
  const GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  Node* node = context.visit_node;
  bool is_root = (node == lookup_table.get_node(root_info.node_index));
  PuctCalculator action_selector(lookup_table, manager_params, search_params, node, is_root);

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

  Derived::print_action_selection_details(context, action_selector, argmax_index);
  return argmax_index;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::load_evaluations(SearchContext& context) {
  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto& stats = node->stats();

    int n = stable_data.num_valid_actions;
    GameResultTensor R;

    LocalPolicyArray P_raw(n);
    LocalActionValueArray child_V(n);

    auto eval = item.eval();

    // assumes that heads are in order policy, value, action-value
    //
    // TODO: we should be able to verify this assumption at compile-time
    std::copy_n(eval->data(0), P_raw.size(), P_raw.data());
    std::copy_n(eval->data(1), R.size(), R.data());
    std::copy_n(eval->data(2), child_V.size(), child_V.data());

    LocalPolicyArray P_adjusted = P_raw;
    Derived::transform_policy(context, P_adjusted);

    stable_data.R = R;
    stable_data.VT_valid = true;

    // No need to worry about thread-safety when modifying edges or stats below, since no other
    // threads can access this node until after load_eval() returns
    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->policy_prior_prob = P_raw[i];
      edge->adjusted_base_prob = P_adjusted[i];
      edge->child_V_estimate = child_V[i];
    }

    ValueArray V = Game::GameResults::to_value_array(R);
    Derived::update_q(stats, V);
    stats.Q_sq = V * V;

    eigen_util::debug_assert_is_valid_prob_distr(V);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_results(const GeneralContext& general_context,
                                                 SearchResults& results) {
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
  for (core::action_t action : stable_data.valid_action_mask.on_indices()) {
    Symmetries::apply(action, inv_sym, mode);
    results.valid_actions.set(action, true);
    actions[i] = action;

    auto* edge = lookup_table.get_edge(root, i);
    results.policy_prior(action) = edge->policy_prior_prob;

    i++;
  }

  Derived::load_action_symmetries(general_context, root, &actions[0], results);
  Derived::write_results(general_context, root, inv_sym, results);
  results.policy_target = results.counts;
  results.provably_lost = stats.provably_losing[stable_data.active_seat];
  results.trivial = root->trivial();
  if (manager_params.forced_playouts && root_info.add_noise) {
    Derived::prune_policy_target(inv_sym, general_context, results);
  }

  Symmetries::apply(results.counts, inv_sym, mode);
  Symmetries::apply(results.policy_target, inv_sym, mode);
  Symmetries::apply(results.Q, inv_sym, mode);
  Symmetries::apply(results.Q_sq, inv_sym, mode);
  Symmetries::apply(results.action_values, inv_sym, mode);

  results.win_rates = stats.Q;
  results.value_prior = stable_data.R;
  results.action_mode = mode;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::write_to_training_info(const TrainingInfoParams& params,
                                                             TrainingInfo& training_info) {
  const SearchResults* mcts_results = params.mcts_results;
  bool use_for_training = params.use_for_training;
  bool previous_used_for_training = params.previous_used_for_training;
  core::seat_index_t seat = params.seat;

  training_info.state = params.state;
  training_info.active_seat = seat;
  training_info.action = params.action;
  training_info.use_for_training = use_for_training;

  if (use_for_training || previous_used_for_training) {
    training_info.policy_target_valid =
      Derived::extract_policy_target(mcts_results, training_info.policy_target);
  }
  if (use_for_training) {
    training_info.action_values_target = mcts_results->action_values;
    training_info.action_values_target_valid = true;
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_record(const TrainingInfo& training_info,
                                                GameLogFullRecord& full_record) {
  full_record.position = training_info.state;

  if (training_info.policy_target_valid) {
    full_record.policy_target = training_info.policy_target;
  } else {
    full_record.policy_target.setZero();
  }

  if (training_info.action_values_target_valid) {
    full_record.action_values = training_info.action_values_target;
  } else {
    full_record.action_values.setZero();
  }

  full_record.action = training_info.action;
  full_record.active_seat = training_info.active_seat;
  full_record.use_for_training = training_info.use_for_training;
  full_record.policy_target_valid = training_info.policy_target_valid;
  full_record.action_values_valid = training_info.action_values_target_valid;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::serialize_record(const GameLogFullRecord& full_record,
                                                       std::vector<char>& buf) {
  GameLogCompactRecord compact_record;
  compact_record.position = full_record.position;
  compact_record.active_seat = full_record.active_seat;
  compact_record.action_mode = Game::Rules::get_action_mode(full_record.position);
  compact_record.action = full_record.action;

  TensorData policy(full_record.policy_target_valid, full_record.policy_target);
  TensorData action_values(full_record.action_values_valid, full_record.action_values);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values.write_to(buf);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::to_view(const GameLogViewParams& params, GameLogView& view) {
  const GameLogCompactRecord* record = params.record;
  const GameLogCompactRecord* next_record = params.next_record;
  const State* cur_pos = params.cur_pos;
  const State* final_pos = params.final_pos;
  const GameResultTensor* outcome = params.outcome;
  group::element_t sym = params.sym;

  core::seat_index_t active_seat = record->active_seat;
  core::action_mode_t mode = record->action_mode;

  const char* addr = reinterpret_cast<const char*>(record);

  const char* policy_data_addr = addr + sizeof(GameLogCompactRecord);
  const TensorData* policy_data = reinterpret_cast<const TensorData*>(policy_data_addr);

  const char* action_values_data_addr = policy_data_addr + policy_data->size();
  const TensorData* action_values_data =
    reinterpret_cast<const TensorData*>(action_values_data_addr);

  view.policy_valid = policy_data->load(view.policy);
  view.action_values_valid = action_values_data->load(view.action_values);

  if (view.policy_valid) {
    Game::Symmetries::apply(view.policy, sym, mode);
  }

  if (view.action_values_valid) {
    Game::Symmetries::apply(view.action_values, sym, mode);
  }

  view.next_policy_valid = false;
  if (next_record) {
    const char* next_addr = reinterpret_cast<const char*>(next_record);

    const char* next_policy_data_addr = next_addr + sizeof(GameLogCompactRecord);
    const TensorData* next_policy_data = reinterpret_cast<const TensorData*>(next_policy_data_addr);

    view.next_policy_valid = next_policy_data->load(view.next_policy);
    if (view.next_policy_valid) {
      Game::Symmetries::apply(view.next_policy, sym, next_record->action_mode);
    }
  }

  view.cur_pos = *cur_pos;
  view.final_pos = *final_pos;
  view.game_result = *outcome;
  view.active_seat = active_seat;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::print_visit_info(const SearchContext& context) {
  if (search::kEnableSearchDebug) {
    const Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), context.search_path_str(),
             node->stable_data().active_seat);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::print_mcts_results(std::ostream& ss,
                                                         const PolicyTensor& action_policy,
                                                         const SearchResults& results,
                                                         int n_rows_to_display) {
  std::cout << std::endl << "CPU pos eval:" << std::endl;
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;
  core::action_mode_t mode = results.action_mode;

  eigen_util::PrintArrayFormatMap fmt_map{
    {"Player", [&](core::seat_index_t x) { return IO::player_to_str(x); }},
    {"action", [&](float x) { return IO::action_to_str(x, mode); }},
  };

  Game::GameResults::print_array(net_value, win_rates, &fmt_map);

  if (Game::Rules::is_chance_mode(results.action_mode)) return;

  int num_valid = valid_actions.count();

  // Zero() calls: not necessary, but silences gcc warning, and is cheap enough
  LocalPolicyArray actions_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray net_policy_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray action_policy_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray mcts_counts_arr = LocalPolicyArray::Zero(num_valid);
  LocalPolicyArray posterior_arr = LocalPolicyArray::Zero(num_valid);

  float total_count = 0;
  for (int a : valid_actions.on_indices()) {
    total_count += mcts_counts(a);
  }

  int r = 0;
  for (int a : valid_actions.on_indices()) {
    actions_arr(r) = a;
    net_policy_arr(r) = net_policy(a);
    action_policy_arr(r) = action_policy(a);
    mcts_counts_arr(r) = mcts_counts(a);
    r++;
  }

  int num_rows = std::min(num_valid, n_rows_to_display);

  posterior_arr = mcts_counts_arr / total_count;
  static std::vector<std::string> columns = {"action", "Prior", "Posterior", "Counts", "Modified"};
  auto data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions_arr, net_policy_arr, posterior_arr, mcts_counts_arr,
                                    action_policy_arr),
    3, false);

  eigen_util::print_array(std::cout, data.topRows(num_rows), columns, &fmt_map);

  if (num_valid > num_rows) {
    int x = num_valid - num_rows;
    if (x == 1) {
      std::cout << "... 1 row not displayed" << std::endl;
    } else {
      std::cout << "... " << x << " rows not displayed" << std::endl;
    }
  } else {
    for (int i = 0; i < n_rows_to_display - num_rows + 1; i++) {
      std::cout << std::endl;
    }
  }
  std::cout << "******************************" << std::endl;
}

template <search::concepts::Traits Traits, typename Derived>
template <typename MutexProtectedFunc>
void AlgorithmsBase<Traits, Derived>::backprop_helper(Node* node, LookupTable& lookup_table,
                                                      MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  func();
  lock.unlock();

  Derived::update_stats(node, lookup_table);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::init_q(NodeStats& stats, const ValueArray& value, bool pure) {
  stats.init_q(value, pure);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_q(NodeStats& stats, const ValueArray& value) {
  stats.update_q(value);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::update_stats(Node* node, LookupTable& lookup_table) {
  ValueArray Q_sum;
  ValueArray Q_sq_sum;
  Q_sum.setZero();
  Q_sq_sum.setZero();
  int N = 0;

  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;
  all_provably_winning.set();
  all_provably_losing.set();

  auto& stats = node->stats();
  const auto& stable_data = node->stable_data();

  int num_valid_actions = stable_data.num_valid_actions;
  core::seat_index_t seat = stable_data.active_seat;

  if (stable_data.is_chance_node) {
    for (int i = 0; i < num_valid_actions; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      const Node* child = lookup_table.get_node(edge->child_index);

      if (!child) {
        break;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      Q_sum += child_stats.Q * edge->chance_prob;
      Q_sq_sum += child_stats.Q_sq * edge->chance_prob;
      N++;

      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
    }
    if (N == num_valid_actions) {
      mit::unique_lock lock(node->mutex());
      Derived::update_q(stats, Q_sum);
      stats.Q_sq = Q_sq_sum;
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }
  } else {
    // provably winning/losing calculation
    bool cp_has_winning_move = false;
    int num_children = 0;

    bool skipped = false;
    for (int i = 0; i < num_valid_actions; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      const Node* child = lookup_table.get_node(edge->child_index);
      if (!child) {
        skipped = true;
        continue;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      if (child_stats.RN > 0) {
        int e = edge->E;
        N += e;
        Q_sum += child_stats.Q * e;
        Q_sq_sum += child_stats.Q_sq * e;
      }

      cp_has_winning_move |= child_stats.provably_winning[seat];
      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
      num_children++;
    }

    if (skipped) {
      all_provably_winning.reset();
      all_provably_losing.reset();
    }

    if (stable_data.VT_valid) {
      ValueArray V = Game::GameResults::to_value_array(stable_data.R);
      Q_sum += V;
      Q_sq_sum += V * V;
      N++;

      eigen_util::debug_assert_is_valid_prob_distr(V);
    }

    auto Q = N ? (Q_sum / N) : Q_sum;
    auto Q_sq = N ? (Q_sq_sum / N) : Q_sq_sum;

    mit::unique_lock lock(node->mutex());
    Derived::update_q(stats, Q);
    stats.Q_sq = Q_sq;
    stats.update_provable_bits(all_provably_winning, all_provably_losing, num_children,
                               cp_has_winning_move, num_valid_actions, seat);

    if (N) {
      eigen_util::debug_assert_is_valid_prob_distr(stats.Q);
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::write_results(const GeneralContext& general_context,
                                                    const Node* root, group::element_t inv_sym,
                                                    SearchResults& results) {
  // This should only be called in contexts where the search-threads are inactive, so we do not need
  // to worry about thread-safety

  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& params = general_context.manager_params;

  core::seat_index_t seat = root->stable_data().active_seat;
  DEBUG_ASSERT(seat >= 0 && seat < kNumPlayers);

  auto& counts = results.counts;
  auto& action_values = results.action_values;
  auto& Q = results.Q;
  auto& Q_sq = results.Q_sq;

  counts.setZero();
  action_values.setZero();
  Q.setZero();
  Q_sq.setZero();

  const auto& parent_stats = root->stats();  // thread-safe because single-threaded here

  bool provably_winning = parent_stats.provably_winning[seat];
  bool provably_losing = parent_stats.provably_losing[seat];

  for (int i = 0; i < root->stable_data().num_valid_actions; i++) {
    const Edge* edge = lookup_table.get_edge(root, i);
    core::action_t action = edge->action;

    int count = edge->E;
    int modified_count = count;

    const Node* child = lookup_table.get_node(edge->child_index);
    if (!child) continue;

    // not actually unsafe since single-threaded
    const auto& child_stats = child->stats();  // thread-safe because single-threaded here
    if (params.avoid_proven_losers && !provably_losing && child_stats.provably_losing[seat]) {
      modified_count = 0;
    } else if (params.exploit_proven_winners && provably_winning &&
               !child_stats.provably_winning[seat]) {
      modified_count = 0;
    }

    if (modified_count) {
      counts(action) = modified_count;
      Q(action) = child_stats.Q(seat);
      Q_sq(action) = child_stats.Q_sq(seat);
    }

    const auto& stable_data = child->stable_data();
    RELEASE_ASSERT(stable_data.VT_valid);
    ValueArray V = Game::GameResults::to_value_array(stable_data.R);
    action_values(action) = V(seat);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::validate_state(LookupTable& lookup_table, Node* node) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (node->is_terminal()) return;

  mit::unique_lock lock(node->mutex());

  int N = 1;
  for (int i = 0; i < node->stable_data().num_valid_actions; ++i) {
    auto edge = lookup_table.get_edge(node, i);
    N += edge->E;
    DEBUG_ASSERT(edge->E >= 0);
  }

  const auto stats_copy = node->stats();  // thread-safe because we hold the mutex
  lock.unlock();

  DEBUG_ASSERT(N == stats_copy.RN + stats_copy.VN, "[{}] {} != {} + {}", (void*)node, N,
               stats_copy.RN, stats_copy.VN);
  DEBUG_ASSERT(stats_copy.RN >= 0);
  DEBUG_ASSERT(stats_copy.VN >= 0);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::transform_policy(SearchContext& context,
                                                       LocalPolicyArray& P) {
  core::node_pool_index_t index = context.initialization_index;
  GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const ManagerParams& manager_params = general_context.manager_params;

  if (index == root_info.node_index) {
    if (search_params.full_search) {
      if (manager_params.dirichlet_mult) {
        Derived::add_dirichlet_noise(general_context, P);
      }
      P = P.pow(1.0 / general_context.aux_state.root_softmax_temperature.value());
      P /= P.sum();
    }
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::add_dirichlet_noise(GeneralContext& general_context,
                                                          LocalPolicyArray& P) {
  const ManagerParams& manager_params = general_context.manager_params;
  auto& dirichlet_gen = general_context.aux_state.dirichlet_gen;
  auto& rng = general_context.aux_state.rng;

  int n = P.rows();
  double alpha = manager_params.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen.template generate<LocalPolicyArray>(rng, alpha, n);
  P = (1.0 - manager_params.dirichlet_mult) * P + manager_params.dirichlet_mult * noise;
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::load_action_symmetries(const GeneralContext& general_context,
                                                             const Node* root,
                                                             core::action_t* actions,
                                                             SearchResults& results) {
  const auto& stable_data = root->stable_data();
  const LookupTable& lookup_table = general_context.lookup_table;

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_actions);

  for (int e = 0; e < stable_data.num_valid_actions; ++e) {
    Edge* edge = lookup_table.get_edge(root, e);
    if (edge->child_index < 0) continue;
    items.emplace_back(edge->child_index, actions[e]);
  }

  results.action_symmetry_table.load(items);
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::prune_policy_target(group::element_t inv_sym,
                                                          const GeneralContext& general_context,
                                                          SearchResults& results) {
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  if (manager_params.no_model) return;

  const Node* root = lookup_table.get_node(root_info.node_index);
  PuctCalculator action_selector(lookup_table, manager_params, search_params, root, true);

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
  auto sqrt_mE = sqrt(mE_sum + PuctCalculator::eps);

  LocalPolicyArray mE_floor = manager_params.cPUCT * P * sqrt_mE / (PUCT_max - 2 * Q) - 1;

  int n_actions = root->stable_data().num_valid_actions;
  for (int i = 0; i < n_actions; ++i) {
    const Edge* edge = lookup_table.get_edge(root, i);
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
      core::action_t raw_action = lookup_table.get_edge(root, i)->action;
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

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::validate_search_path(const SearchContext& context) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  LookupTable& lookup_table = context.general_context->lookup_table;
  int N = context.search_path.size();
  for (int i = N - 1; i >= 0; --i) {
    Derived::validate_state(lookup_table, context.search_path[i].node);
  }
}

template <search::concepts::Traits Traits, typename Derived>
void AlgorithmsBase<Traits, Derived>::print_action_selection_details(const SearchContext& context,
                                                                     const PuctCalculator& selector,
                                                                     int argmax_index) {
  LookupTable& lookup_table = context.general_context->lookup_table;
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
      auto edge = lookup_table.get_edge(node, e);
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

template <search::concepts::Traits Traits, typename Derived>
bool AlgorithmsBase<Traits, Derived>::extract_policy_target(const SearchResults* mcts_results,
                                                            PolicyTensor& target) {
  target = mcts_results->policy_target;

  float sum = eigen_util::sum(target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    // python training code will ignore these rows for policy training.
    return false;
  } else {
    target = mcts_results->action_symmetry_table.symmetrize(target);
    target = target / eigen_util::sum(target);
    return true;
  }
}

}  // namespace alpha0
