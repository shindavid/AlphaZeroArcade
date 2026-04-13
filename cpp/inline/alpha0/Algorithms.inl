#include "alpha0/Algorithms.hpp"

#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "util/Asserts.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"
#include "util/LoggingUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <boost/algorithm/string/join.hpp>
#include <spdlog/spdlog.h>

#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace alpha0 {

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::print_visit_info(const SearchContext& context) {
  if (search::kEnableSearchDebug) {
    const Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), context.search_path_str(),
             node->stable_data().active_seat);
  }
}

template <search::concepts::SearchSpec SearchSpec>
bool Algorithms<SearchSpec>::validate_and_symmetrize_policy_target(
  const SearchResults* mcts_results, PolicyTensor& target) {
  float sum = eigen_util::sum(target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    // python training code will ignore these rows for policy training.
    return false;
  } else {
    target = mcts_results->action_symmetry_table.symmetrize(mcts_results->frame, target);
    target = target / eigen_util::sum(target);
    eigen_util::debug_assert_is_valid_prob_distr(target);
    return true;
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::load_action_symmetries(const GeneralContext& general_context,
                                                    const Node* root, SearchResults& results) {
  const auto& stable_data = root->stable_data();
  const LookupTable& lookup_table = general_context.lookup_table;
  const State& root_state = general_context.root_info.state;

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_moves);

  using equivalence_class_t = int;
  using map_t = std::unordered_map<InputFrame, equivalence_class_t>;
  map_t map;

  State state = root_state;  // copy
  for (int e = 0; e < stable_data.num_valid_moves; ++e) {
    Edge* edge = lookup_table.get_edge(root, e);
    Game::Rules::apply(state, edge->move);
    InputFrame frame(state);
    group::element_t sym = Symmetries::get_canonical_symmetry(frame);
    Symmetries::apply(frame, sym);

    auto [it, inserted] = map.try_emplace(frame, map.size());
    items.emplace_back(it->second, edge->move);
    Game::Rules::backtrack_state(state, root_state);
  }

  results.action_symmetry_table.load(items);
  results.trivial = (map.size() <= 1);
}

template <search::concepts::SearchSpec SearchSpec>
typename Algorithms<SearchSpec>::ActionValueTensor Algorithms<SearchSpec>::apply_mask(
  const ActionValueTensor& values, const PolicyTensor& mask, float invalid_value) {
  using Indices = Eigen::array<Eigen::Index, 2>;
  Indices reshape_dims = {mask.dimensions()[0], 1};
  Indices bcast = {1, values.dimensions()[1]};
  auto reshaped_mask = mask.reshape(reshape_dims).broadcast(bcast);
  auto selector = reshaped_mask > reshaped_mask.constant(0.5f);
  ActionValueTensor invalid_tensor = reshaped_mask.constant(invalid_value);
  return selector.select(values, invalid_tensor);
}

template <search::concepts::SearchSpec SearchSpec>
template <typename MutexProtectedFunc>
void Algorithms<SearchSpec>::backprop(SearchContext& context, Node* node, Edge* edge,
                                      MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  func();
  if (!edge) return;
  NodeStats stats = node->stats();  // copy
  lock.unlock();

  update_stats(stats, node, context.general_context->lookup_table);

  lock.lock();

  // Carefully copy back fields of stats back to node->stats()
  // We don't copy counts, which may have been updated by other threads.
  int RN = node->stats().RN;
  int VN = node->stats().VN;
  node->stats() = stats;
  node->stats().RN = RN;
  node->stats().VN = VN;
  lock.unlock();
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::init_node_stats_from_terminal(Node* node) {
  NodeStats& stats = node->stats();
  RELEASE_ASSERT(stats.RN == 0);
  const ValueArray q = node->stable_data().V();

  stats.Q = q;
  stats.Q_sq = q * q;

  for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
    stats.provably_winning[p] = q(p) >= GameResultEncoding::kMaxValue;
    stats.provably_losing[p] = q(p) <= GameResultEncoding::kMinValue;
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::update_node_stats(Node* node, bool undo_virtual) {
  auto& stats = node->stats();

  stats.RN++;
  stats.VN -= undo_virtual;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual) {
  auto& stats = node->stats();

  edge->E += !undo_virtual;
  stats.RN++;
  stats.VN -= undo_virtual;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::virtually_update_node_stats(Node* node) {
  node->stats().VN++;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::virtually_update_node_stats_and_edge(Node* node, Edge* edge) {
  edge->E++;
  node->stats().VN++;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::undo_virtual_update(Node* node, Edge* edge) {
  edge->E--;
  node->stats().VN--;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::validate_search_path(const SearchContext& context) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;

  LookupTable& lookup_table = context.general_context->lookup_table;
  int N = context.search_path.size();
  for (int i = N - 1; i >= 0; --i) {
    validate_state(lookup_table, context.search_path[i].node);
  }
}

template <search::concepts::SearchSpec SearchSpec>
bool Algorithms<SearchSpec>::should_short_circuit(const Edge* edge, const Node* child) {
  int edge_count = edge->E;
  int child_count = child->stats().RN;  // not thread-safe but race-condition is benign
  return edge_count < child_count;
}

template <search::concepts::SearchSpec SearchSpec>
bool Algorithms<SearchSpec>::more_search_iterations_needed(const GeneralContext& general_context,
                                                           const Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  const search::SearchParams& search_params = general_context.search_params;
  if (!search_params.ponder && root->stable_data().num_valid_moves == 1) return false;
  return root->stats().total_count() <= search_params.tree_size_limit;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::init_root_info(GeneralContext& general_context,
                                            search::RootInitPurpose purpose) {
  const ManagerParams& manager_params = general_context.manager_params;
  const search::SearchParams& search_params = general_context.search_params;

  bool add_noise = false;
  switch (purpose) {
    case search::kForStandardSearch: {
      add_noise = search_params.full_search && manager_params.mode == search::kTraining;
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

    const State& cur_state = root_info.state;
    core::seat_index_t active_seat = Game::Rules::get_current_player(cur_state);
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info.active_seat = active_seat;
    auto legal_moves = Game::Rules::analyze(cur_state).valid_moves();
    new (root) Node(lookup_table.get_random_mutex(), cur_state, legal_moves.size(), active_seat);
  }

  if (search::kEnableSearchDebug && purpose == search::kForStandardSearch) {
    IO::print_state(std::cout, root_info.state);
  }
}

template <search::concepts::SearchSpec SearchSpec>
int Algorithms<SearchSpec>::get_best_child_index(const SearchContext& context) {
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

  print_action_selection_details(context, action_selector, argmax_index);
  return argmax_index;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::load_evaluations(SearchContext& context) {
  const LookupTable& lookup_table = context.general_context->lookup_table;
  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto& stats = node->stats();

    int n = stable_data.num_valid_moves;
    GameResultTensor R;

    LocalPolicyArray P_raw(n);
    LocalActionValueArray AV(n, Game::Constants::kNumPlayers);

    auto eval = item.eval();

    using NetworkHeadsList = SearchSpec::EvalSpec::NetworkHeads::List;
    using Head0 = mp::TypeAt_t<NetworkHeadsList, 0>;
    using Head1 = mp::TypeAt_t<NetworkHeadsList, 1>;
    using Head2 = mp::TypeAt_t<NetworkHeadsList, 2>;

    static_assert(util::str_equal<Head0::kName, "policy">());
    static_assert(util::str_equal<Head1::kName, "value">());
    static_assert(util::str_equal<Head2::kName, "action_value">());

    std::copy_n(eval->data(0), P_raw.size(), P_raw.data());
    std::copy_n(eval->data(1), R.size(), R.data());
    std::copy_n(eval->data(2), AV.size(), AV.data());

    RELEASE_ASSERT(eigen_util::isfinite(P_raw), "Non-finite values in policy head");
    RELEASE_ASSERT(eigen_util::isfinite(R), "Non-finite values in value head");
    RELEASE_ASSERT(eigen_util::isfinite(AV), "Non-finite values in action value head");

    LocalPolicyArray P_adjusted = P_raw;
    transform_policy(context, P_adjusted);

    stable_data.R = R;
    stable_data.R_valid = true;

    // No need to worry about thread-safety when modifying edges or stats below, since no other
    // threads can access this node until after load_eval() returns
    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->policy_prior_prob = P_raw[i];
      edge->adjusted_base_prob = P_adjusted[i];
      edge->child_AV = AV.row(i);
    }

    ValueArray V = GameResultEncoding::to_value_array(R);
    stats.Q = V;
    stats.Q_sq = V * V;
  }

  const RootInfo& root_info = context.general_context->root_info;
  Node* root = lookup_table.get_node(root_info.node_index);
  if (root) {
    root->stats().RN = std::max(root->stats().RN, 1);
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::to_results(const GeneralContext& general_context,
                                        SearchResults& results) {
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  const Node* root = lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here
  const State& state = root_info.state;

  results.valid_moves = Game::Rules::analyze(state).valid_moves();
  results.frame = root_info.input_encoder.current_frame();
  results.P.setZero();
  results.pre_expanded_moves.setZero();

  RELEASE_ASSERT((int)results.valid_moves.size() == stable_data.num_valid_moves, "{} != {}",
                 results.valid_moves.size(), stable_data.num_valid_moves);

  int i = 0;
  for (Move move : results.valid_moves) {
    auto* edge = lookup_table.get_edge(root, i);
    auto index = PolicyEncoding::to_index(results.frame, move);
    results.P.coeffRef(index) = edge->policy_prior_prob;
    results.pre_expanded_moves.coeffRef(index) = edge->was_pre_expanded;

    i++;
  }

  load_action_symmetries(general_context, root, results);
  write_results(general_context, root, results);
  results.policy_target = results.counts;
  results.provably_lost = stats.provably_losing[stable_data.active_seat];
  if (manager_params.forced_playouts && root_info.add_noise) {
    prune_policy_target(general_context, results);
  }

  results.Q = stats.Q;
  results.R = stable_data.R;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::write_to_training_info(
  bool use_for_training, const ActionResponse& response, const SearchResults* mcts_results,
  core::seat_index_t seat, GameWriteLog_sptr game_log, TrainingInfo& training_info) {
  // TODO: if we have chance-events between player-events, we should compute this bool
  // differently.
  bool previous_used_for_training = game_log->was_previous_entry_used_for_policy_training();

  training_info.clear();
  training_info.frame = mcts_results->frame;
  training_info.active_seat = seat;
  training_info.move = response.get_move();
  training_info.use_for_training = use_for_training;

  if (use_for_training || previous_used_for_training) {
    training_info.policy_target = mcts_results->policy_target;
    training_info.policy_target_valid =
      validate_and_symmetrize_policy_target(mcts_results, training_info.policy_target);
  }

  if (use_for_training) {
    training_info.action_values_target =
      apply_mask(mcts_results->AV, mcts_results->pre_expanded_moves);
    training_info.action_values_target_valid = true;
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::to_record(const TrainingInfo& training_info,
                                       GameLogFullRecord& full_record) {
  full_record.frame = training_info.frame;

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

  full_record.move = training_info.move;
  full_record.active_seat = training_info.active_seat;
  full_record.use_for_training = training_info.use_for_training;
  full_record.policy_target_valid = training_info.policy_target_valid;
  full_record.action_values_valid = training_info.action_values_target_valid;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::serialize_record(const GameLogFullRecord& full_record,
                                              std::vector<char>& buf) {
  GameLogCompactRecord compact_record;
  compact_record.frame = full_record.frame;
  compact_record.active_seat = full_record.active_seat;
  compact_record.move = full_record.move;

  PolicyTensorData policy(full_record.policy_target_valid, full_record.policy_target);
  ActionValueTensorData action_values(full_record.action_values_valid, full_record.action_values);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values.write_to(buf);
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::to_view(const GameLogViewParams& params, GameLogView& view) {
  const GameLogCompactRecord* record = params.record;
  const GameLogCompactRecord* next_record = params.next_record;
  const InputFrame* cur_frame = params.cur_frame;
  const InputFrame* final_frame = params.final_frame;
  const GameResultTensor* outcome = params.outcome;
  group::element_t sym = params.sym;

  core::seat_index_t active_seat = record->active_seat;

  const char* addr = reinterpret_cast<const char*>(record);

  const char* policy_data_addr = addr + sizeof(GameLogCompactRecord);
  const PolicyTensorData* policy_data = reinterpret_cast<const PolicyTensorData*>(policy_data_addr);

  const char* action_values_data_addr = policy_data_addr + policy_data->size();
  const ActionValueTensorData* action_values_data =
    reinterpret_cast<const ActionValueTensorData*>(action_values_data_addr);

  view.policy_valid = policy_data->load(view.policy);
  view.action_values_valid = action_values_data->load(view.action_values);

  if (view.policy_valid) {
    Symmetries::apply(view.policy, sym, *cur_frame);
  }

  if (view.action_values_valid) {
    Symmetries::apply(view.action_values, sym, *cur_frame);
  }

  view.next_policy_valid = false;
  if (next_record) {
    const char* next_addr = reinterpret_cast<const char*>(next_record);

    const char* next_policy_data_addr = next_addr + sizeof(GameLogCompactRecord);
    const PolicyTensorData* next_policy_data =
      reinterpret_cast<const PolicyTensorData*>(next_policy_data_addr);

    view.next_policy_valid = next_policy_data->load(view.next_policy);
    if (view.next_policy_valid) {
      Symmetries::apply(view.next_policy, sym, next_record->frame);
    }
  }

  view.cur_frame = *cur_frame;
  view.final_frame = *final_frame;
  view.game_result = *outcome;
  view.active_seat = active_seat;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::update_stats(NodeStats& stats, const Node* node,
                                          LookupTable& lookup_table) {
  ValueArray Q_sum;
  ValueArray Q_sq_sum;
  Q_sum.setZero();
  Q_sq_sum.setZero();

  player_bitset_t all_provably_winning;
  player_bitset_t all_provably_losing;
  all_provably_winning.set();
  all_provably_losing.set();

  const auto& stable_data = node->stable_data();

  int num_valid_moves = stable_data.num_valid_moves;
  core::seat_index_t seat = stable_data.active_seat;

  if (stable_data.is_chance_node) {
    int num_expanded_edges = 0;
    for (int i = 0; i < num_valid_moves; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      const Node* child = lookup_table.get_node(edge->child_index);

      if (!child) {
        break;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      Q_sum += child_stats.Q * edge->chance_prob;
      Q_sq_sum += child_stats.Q_sq * edge->chance_prob;
      num_expanded_edges++;

      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
    }
    if (num_expanded_edges == num_valid_moves) {
      stats.Q = Q_sum;
      stats.Q_sq = Q_sq_sum;
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }
    return;
  } else {
    // provably winning/losing calculation
    bool cp_has_winning_move = false;
    int num_expanded_edges = 0;
    int N = 0;

    DEBUG_ASSERT(num_valid_moves > 0);
    for (int i = 0; i < num_valid_moves; i++) {
      const Edge* edge = lookup_table.get_edge(node, i);
      const Node* child = lookup_table.get_node(edge->child_index);
      if (!child) {
        continue;
      }
      const auto child_stats = child->stats_safe();  // make a copy
      if (child_stats.RN > 0) {
        int e = edge->E;
        N += e;
        Q_sum += child_stats.Q * e;
        Q_sq_sum += child_stats.Q_sq * e;
        eigen_util::debug_assert_is_valid_prob_distr(child_stats.Q);
      }

      cp_has_winning_move |= child_stats.provably_winning[seat];
      all_provably_winning &= child_stats.provably_winning;
      all_provably_losing &= child_stats.provably_losing;
      num_expanded_edges++;
    }

    bool all_edges_expanded = (num_expanded_edges == num_valid_moves);
    if (!all_edges_expanded) {
      all_provably_winning.reset();
      all_provably_losing.reset();
    }

    DEBUG_ASSERT(stable_data.R_valid);
    ValueArray V = stable_data.V();
    Q_sum += V;
    Q_sq_sum += V * V;
    N++;
    eigen_util::debug_assert_is_valid_prob_distr(V);

    auto Q = Q_sum / N;
    auto Q_sq = Q_sq_sum / N;

    stats.Q = Q;
    stats.Q_sq = Q_sq;
    eigen_util::debug_assert_is_valid_prob_distr(stats.Q);
    if (cp_has_winning_move) {
      stats.provably_winning[seat] = true;
      stats.provably_losing.set();
      stats.provably_losing[seat] = false;
    } else if (all_edges_expanded) {
      stats.provably_winning = all_provably_winning;
      stats.provably_losing = all_provably_losing;
    }
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::write_results(const GeneralContext& general_context, const Node* root,
                                           SearchResults& results) {
  // This should only be called in contexts where the search-threads are inactive, so we do not need
  // to worry about thread-safety

  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& params = general_context.manager_params;

  core::seat_index_t seat = root->stable_data().active_seat;
  DEBUG_ASSERT(seat >= 0 && seat < kNumPlayers);

  const auto& frame = results.frame;
  auto& counts = results.counts;
  auto& AV = results.AV;
  auto& AQs = results.AQs;
  auto& AQs_sq = results.AQs_sq;

  counts.setZero();
  AV.setZero();
  AQs.setZero();
  AQs_sq.setZero();

  const auto& parent_stats = root->stats();  // thread-safe because single-threaded here

  bool provably_winning = parent_stats.provably_winning[seat];
  bool provably_losing = parent_stats.provably_losing[seat];

  for (int i = 0; i < root->stable_data().num_valid_moves; i++) {
    const Edge* edge = lookup_table.get_edge(root, i);
    Move move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);

    int count = edge->E;
    int modified_count = count;

    const Node* child = lookup_table.get_node(edge->child_index);
    if (!child) continue;

    const auto& child_stats = child->stats();  // thread-safe because single-threaded here
    if (params.avoid_proven_losers && !provably_losing && child_stats.provably_losing[seat]) {
      modified_count = 0;
    } else if (params.exploit_proven_winners && provably_winning &&
               !child_stats.provably_winning[seat]) {
      modified_count = 0;
    }

    if (modified_count) {
      counts.coeffRef(index) = modified_count;
      AQs.coeffRef(index) = child_stats.Q(seat);
      AQs_sq.coeffRef(index) = child_stats.Q_sq(seat);
    }

    const auto& stable_data = child->stable_data();
    RELEASE_ASSERT(stable_data.R_valid);
    ValueArray V = stable_data.V();
    eigen_util::chip_assign(AV, eigen_util::reinterpret_as_tensor(V), index);
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::validate_state(LookupTable& lookup_table, Node* node) {
  if (!IS_DEFINED(DEBUG_BUILD)) return;
  if (node->is_terminal()) return;

  mit::unique_lock lock(node->mutex());

  int N = 1;
  for (int i = 0; i < node->stable_data().num_valid_moves; ++i) {
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

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::transform_policy(SearchContext& context, LocalPolicyArray& P) {
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
      float temp = general_context.aux_state.root_softmax_temperature.value();
      if (temp > 0.0f) {
        P = P.pow(1.0f / temp);
      }
      P /= P.sum();
    }
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::add_dirichlet_noise(GeneralContext& general_context,
                                                 LocalPolicyArray& P) {
  const ManagerParams& manager_params = general_context.manager_params;
  auto& dirichlet_gen = general_context.aux_state.dirichlet_gen;
  auto& rng = general_context.aux_state.rng;

  int n = P.rows();
  double alpha = manager_params.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen.template generate<LocalPolicyArray>(rng, alpha, n);
  P = (1.0 - manager_params.dirichlet_mult) * P + manager_params.dirichlet_mult * noise;
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::prune_policy_target(const GeneralContext& general_context,
                                                 SearchResults& results) {
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  if (manager_params.no_model) return;

  const auto& frame = results.frame;
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
  auto denom = PUCT_max - 2 * Q;

  LocalPolicyArray mE_floor = manager_params.cPUCT * P * sqrt_mE / denom - 1;

  int n_moves = root->stable_data().num_valid_moves;
  for (int i = 0; i < n_moves; ++i) {
    const Edge* edge = lookup_table.get_edge(root, i);
    const Move& move = edge->move;
    auto index = PolicyEncoding::to_index(frame, move);
    if (mE(i) == 0) {
      results.policy_target.coeffRef(index) = 0;
      continue;
    }
    if (mE(i) == mE_max) continue;
    if (denom(i) == 0) continue;
    if (mE_floor(i) >= mE(i)) continue;
    auto n = std::max(mE_floor(i), mE(i) - n_forced(i));
    if (n <= 1.0) {
      n = 0;
    }
    results.policy_target.coeffRef(index) = n;
  }

  if (eigen_util::sum(results.policy_target) <= 0) {
    // can happen in certain edge cases
    results.policy_target = results.counts;
  }

  if (search::kEnableSearchDebug) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    LocalPolicyArray pruned(n_moves);

    ActionPrinter printer(lookup_table.get_moves(root));
    for (int i = 0; i < n_moves; ++i) {
      const Edge* edge = lookup_table.get_edge(root, i);
      Move move = edge->move;
      auto index = PolicyEncoding::to_index(frame, move);

      pruned(i) = results.policy_target.coeff(index);
    }

    LocalPolicyArray actions = printer.flat_array();
    LocalPolicyArray target = pruned / pruned.sum();

    static std::vector<std::string> columns = {"action", "P",  "Q",  "PUCT",   "E",
                                               "PW",     "PL", "mE", "pruned", "target"};
    auto data = eigen_util::sort_rows(
      eigen_util::concatenate_columns(actions, P, Q, PUCT, E, PW, PL, mE, pruned, target));

    eigen_util::PrintArrayFormatMap fmt_map;
    printer.update_format_map(fmt_map);

    std::cout << std::endl << "Policy target pruning:" << std::endl;
    eigen_util::print_array(std::cout, data, columns, &fmt_map);
#pragma GCC diagnostic pop
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::print_action_selection_details(const SearchContext& context,
                                                            const PuctCalculator& selector,
                                                            int argmax_index) {
  LookupTable& lookup_table = context.general_context->lookup_table;
  Node* node = context.visit_node;
  if (search::kEnableSearchDebug) {
    std::ostringstream ss;

    core::seat_index_t seat = node->stable_data().active_seat;

    int n_moves = node->stable_data().num_valid_moves;

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
      {"Seat", [&](float x, int) { return std::to_string(int(x)); }},
      {"CurP", [&](float x, int) { return std::string(x ? "*" : ""); }},
    };

    eigen_util::print_array(ss, player_data, player_columns, &fmt_map1);

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

    LocalPolicyArray child_addr(n_moves);
    LocalPolicyArray argmax(n_moves);
    child_addr.setConstant(-1);
    argmax.setZero();
    argmax(argmax_index) = 1;

    ActionPrinter printer(lookup_table.get_moves(node));
    for (int i = 0; i < n_moves; ++i) {
      const Edge* edge = lookup_table.get_edge(node, i);
      child_addr(i) = edge->child_index;
    }

    LocalPolicyArray actions = printer.flat_array();

    static std::vector<std::string> action_columns = {
      "action", "P", "Q", "FPU", "PW", "PL", "E", "mE", "RN", "VN", "&ch", "PUCT", "argmax"};
    auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
      actions, P, Q, FPU, PW, PL, E, mE, RN, VN, child_addr, PUCT, argmax));

    eigen_util::PrintArrayFormatMap fmt_map2{
      {"&ch", [](float x, int) { return x < 0 ? std::string() : std::to_string((int)x); }},
      {"argmax", [](float x, int) { return std::string(x == 0 ? "" : "*"); }},
    };
    printer.update_format_map(fmt_map2);

    eigen_util::print_array(ss, action_data, action_columns, &fmt_map2);
    util::Logging::multi_line_log_info(ss.str(), context.log_prefix_n());
  }
}

}  // namespace alpha0
