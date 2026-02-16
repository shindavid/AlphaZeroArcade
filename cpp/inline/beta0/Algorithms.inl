#include "beta0/Algorithms.hpp"

#include "alpha0/Algorithms.hpp"
#include "beta0/Backpropagator.hpp"
#include "beta0/Calculations.hpp"
#include "beta0/Constants.hpp"
#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "util/EigenUtil.hpp"
#include "util/Math.hpp"
#include "util/MetaProgramming.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep
#include "x0/Algorithms.hpp"

#include <cmath>

namespace beta0 {

template <search::concepts::Traits Traits>
template <typename MutexProtectedFunc>
void Algorithms<Traits>::backprop(SearchContext& context, Node* node, Edge* edge,
                                  MutexProtectedFunc&& func) {
  if (!edge) {
    mit::unique_lock lock(node->mutex());
    func();
    return;
  }

  using Backpropagator = beta0::Backpropagator<Traits>;
  Backpropagator backpropagator(context, node, edge, func);
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::init_node_stats_from_terminal(Node* node) {
  const ValueArray q = node->stable_data().V();

  NodeStats& stats = node->stats();
  RELEASE_ASSERT(stats.N == 0);

  stats.Q = q;
  stats.Q_min = stats.Q;
  stats.Q_max = stats.Q;
  stats.W.fill(0.f);
  Calculations<Game>::populate_logit_value_beliefs(stats.Q, stats.W, stats.lQW, kAllowInf);
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::update_node_stats(Node* node, bool undo_virtual) {
  node->stats().N++;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual) {
  node->stats().N++;
}

template <search::concepts::Traits Traits>
bool Algorithms<Traits>::more_search_iterations_needed(const GeneralContext& general_context,
                                                       const Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  const search::SearchParams& search_params = general_context.search_params;
  if (!search_params.ponder && root->stable_data().num_valid_actions == 1) return false;
  if (root->stats().W.isZero(0.f)) return false;
  return root->stats().N <= search_params.tree_size_limit;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::init_root_info(GeneralContext& general_context,
                                        search::RootInitPurpose purpose) {
  RootInfo& root_info = general_context.root_info;
  LookupTable& lookup_table = general_context.lookup_table;

  if (root_info.node_index < 0) {
    root_info.node_index = lookup_table.alloc_node();
    Node* root = lookup_table.get_node(root_info.node_index);

    const State& cur_state = root_info.input_tensorizor.current_state();
    core::seat_index_t active_seat = Game::Rules::get_current_player(cur_state);
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info.active_seat = active_seat;
    new (root) Node(lookup_table.get_random_mutex(), cur_state, active_seat);
  }

  if (search::kEnableSearchDebug && purpose == search::kForStandardSearch) {
    const auto& state = root_info.input_tensorizor.current_state();
    Game::IO::print_state(std::cout, state);
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::init_root_edges(GeneralContext& general_context) {
  const search::SearchParams& search_params = general_context.search_params;
  const ManagerParams& manager_params = general_context.manager_params;

  if (!search_params.full_search) return;
  if (!manager_params.enable_exploratory_visits) return;

  RootInfo& root_info = general_context.root_info;
  LookupTable& lookup_table = general_context.lookup_table;
  auto& dirichlet_gen = general_context.aux_state.dirichlet_gen;
  auto& rng = general_context.aux_state.rng;

  Node* root = lookup_table.get_node(root_info.node_index);
  int n = root->stable_data().num_valid_actions;

  RELEASE_ASSERT(n > 0);
  int XC[n];
  for (int i = 0; i < n; ++i) {
    XC[i] = 0;
  }
  double alpha = manager_params.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen.template generate<LocalPolicyArray>(rng, alpha, n);
  const float* f = noise.data();
  std::discrete_distribution<int> dist(f, f + n);
  for (int i = 0; i < 30; ++i) {  // TODO: make this 30 configurable
    int a = dist(rng);
    XC[a] += 1;
  }

  for (int i = 0; i < n; ++i) {
    Edge* edge = lookup_table.get_edge(root, i);
    edge->XC = XC[i];
  }
}

template <search::concepts::Traits Traits>
int Algorithms<Traits>::get_best_child_index(const SearchContext& context) {
  // search criterion = pi_i * sqrt(W_i) * (N(p) - RC_i)
  const GeneralContext& general_context = *context.general_context;
  const search::SearchParams& search_params = general_context.search_params;
  const LookupTable& lookup_table = general_context.lookup_table;

  Node* node = context.visit_node;
  core::seat_index_t seat = node->stable_data().active_seat;
  int n = node->stable_data().num_valid_actions;

  using Array = LocalPolicyArray;

  Array W(n);
  Array pi(n);
  Array XC(n);
  Array XM(n);
  Array RC(n);
  Array S(n);
  Array N(n);
  Array score_sq(n);
  int pN;
  bool XM_any = false;

  mit::unique_lock lock(node->mutex());
  pN = node->stats().N + 1;
  for (int i = 0; i < n; ++i) {
    Edge* edge = lookup_table.get_edge(node, i);
    Node* child = lookup_table.get_node(edge->child_index);
    if (child) {
      W(i) = child->stats().W[seat];
      N(i) = child->stats().N;
    } else {
      W(i) = edge->child_AU[seat];
      N(i) = 0;
    }
    pi(i) = edge->pi;
    XC(i) = edge->XC;
    RC(i) = edge->RC;

    bool xc_active = XC(i) > N(i);
    XM(i) = xc_active;
    XM_any |= xc_active;
  }
  lock.unlock();

  S = pN - RC;

  int argmax_index;
  if (search_params.tree_size_limit == 1) {
    // net-only, use pi
    pi.maxCoeff(&argmax_index);
  } else {
    if (XM_any) {
      XM.maxCoeff(&argmax_index);
    } else {
      score_sq = W * pi * pi * S * S;
      score_sq.maxCoeff(&argmax_index);
    }

    if (search::kEnableSearchDebug) {
      Array score(n);
      Array actions(n);
      Array argmax(n);
      Array sqrt_W = W.cwiseSqrt();

      if (XM_any) {
        score = XM;
      } else {
        score = score_sq.cwiseSqrt();
      }

      for (int e = 0; e < n; ++e) {
        auto edge = lookup_table.get_edge(node, e);
        actions(e) = edge->action;
      }
      argmax.setZero();
      argmax(argmax_index) = 1;

      std::ostringstream ss;
      ss << std::format("{:>{}}", "", context.log_prefix_n());

      static std::vector<std::string> action_columns = {"action", "sqrt(W)", "pi", "S",  "score",
                                                        "N",      "RC",      "XC", "XM", "argmax"};
      auto action_data = eigen_util::sort_rows(
        eigen_util::concatenate_columns(actions, sqrt_W, pi, S, score, N, RC, XC, XM, argmax));

      eigen_util::PrintArrayFormatMap fmt_map{
        {"action", [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); }},
        {"argmax", [](float x) { return std::string(x == 0 ? "" : "*"); }},
      };

      std::stringstream ss2;
      eigen_util::print_array(ss2, action_data, action_columns, &fmt_map);

      std::string line_break =
        std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context.log_prefix_n());
      for (const std::string& line : util::splitlines(ss2.str())) {
        ss << line << line_break;
      }

      LOG_INFO(ss.str());
    }
  }

  return argmax_index;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::load_evaluations(SearchContext& context) {
  GeneralContext& general_context = *context.general_context;
  const LookupTable& lookup_table = general_context.lookup_table;
  const RootInfo& root_info = general_context.root_info;
  Node* root = lookup_table.get_node(root_info.node_index);

  for (auto& item : context.eval_request.fresh_items()) {
    Node* node = static_cast<Node*>(item.node());

    auto& stable_data = node->stable_data();
    auto& stats = node->stats();
    auto eval = item.eval();
    auto seat = stable_data.active_seat;

    int n = stable_data.num_valid_actions;

    GameResultTensor R;
    ValueArray U;
    LocalActionValueArray AU(n, kNumPlayers);
    LocalPolicyArray P(n);
    LocalActionValueArray AV(n, kNumPlayers);

    using NetworkHeadsList = Traits::EvalSpec::NetworkHeads::List;
    using Head0 = mp::TypeAt_t<NetworkHeadsList, 0>;
    using Head1 = mp::TypeAt_t<NetworkHeadsList, 1>;
    using Head2 = mp::TypeAt_t<NetworkHeadsList, 2>;
    using Head3 = mp::TypeAt_t<NetworkHeadsList, 3>;
    using Head4 = mp::TypeAt_t<NetworkHeadsList, 4>;

    static_assert(util::str_equal<Head0::kName, "policy">());
    static_assert(util::str_equal<Head1::kName, "value">());
    static_assert(util::str_equal<Head2::kName, "action_value">());
    static_assert(util::str_equal<Head3::kName, "value_uncertainty">());
    static_assert(util::str_equal<Head4::kName, "action_value_uncertainty">());

    std::copy_n(eval->data(0), P.size(), P.data());
    std::copy_n(eval->data(1), R.size(), R.data());
    std::copy_n(eval->data(2), AV.size(), AV.data());
    std::copy_n(eval->data(3), U.size(), U.data());
    std::copy_n(eval->data(4), AU.size(), AU.data());

    RELEASE_ASSERT(eigen_util::isfinite(P), "Non-finite values in policy head");
    RELEASE_ASSERT(eigen_util::isfinite(R), "Non-finite values in value head");
    RELEASE_ASSERT(eigen_util::isfinite(U), "Non-finite values in value uncertainty head");
    RELEASE_ASSERT(eigen_util::isfinite(AV), "Non-finite values in action value head");
    RELEASE_ASSERT(eigen_util::isfinite(AU), "Non-finite values in action value uncertainty head");

    ValueArray V = Game::GameResults::to_value_array(R);

    ValueArray U_original = U;
    LocalActionValueArray AV_original = AV;

    Calculations<Game>::calibrate_priors(seat, P, V, U, AV, AU);

    LocalPolicyArray A = P;
    for (int i = 0; i < n; ++i) {
      A(i) = math::fast_coarse_log_less_than_1(P(i));
    }
    A -= (1.f + A.maxCoeff());  // calibrate A so max A is -1

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->P = P[i];
      edge->A = A[i];
      edge->child_AU = AU.row(i);
      edge->child_AV = AV.row(i);
      edge->pi = edge->P;
      Calculations<Game>::populate_logit_value_beliefs(AV.row(i), AU.row(i), edge->child_lAUV);
      edge->child_lQW = edge->child_lAUV[seat];
    }

    stable_data.R = R;
    stable_data.R_valid = true;
    stable_data.U = U;
    Calculations<Game>::populate_logit_value_beliefs(V, U, stable_data.lUV);

    stats.Q = V;
    stats.lQW = stable_data.lUV;
    stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
    stats.Q_max = stats.Q_max.cwiseMax(stats.Q);
    stats.W = U;

    if (search::kEnableSearchDebug) {
      std::ostringstream ss;
      ss << std::format("{:>{}}", "", context.log_prefix_n());

      std::string line_break =
        std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context.log_prefix_n());

      ss << "NN EVAL" << line_break;
      ValueArray players;
      ValueArray CP;
      ValueArray lU;
      ValueArray lV;
      for (int p = 0; p < kNumPlayers; ++p) {
        players(p) = p;
        CP(p) = p == seat;
        lU(p) = stable_data.lUV[p].variance();
        lV(p) = stable_data.lUV[p].mean();
      }

      static std::vector<std::string> player_columns = {"Seat", "V",  "U_orig", "U",
                                                        "lV",   "lU", "CurP"};
      auto player_data = eigen_util::concatenate_columns(players, V, U_original, U, lV, lU, CP);

      eigen_util::PrintArrayFormatMap fmt_map1{
        {"Seat", [&](float x) { return std::to_string(int(x)); }},
        {"CurP", [&](float x) { return std::string(x ? "*" : ""); }},
      };

      std::stringstream ss1;
      eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

      for (const std::string& line : util::splitlines(ss1.str())) {
        ss << line << line_break;
      }

      ss << line_break;

      LocalPolicyArray actions(n);
      LocalPolicyArray A2(n);
      LocalPolicyArray AVs_original(n);
      LocalPolicyArray AVs(n);
      LocalPolicyArray AUs(n);
      LocalPolicyArray lAVs_original(n);
      LocalPolicyArray lAVs(n);
      LocalPolicyArray lAUs(n);

      for (int e = 0; e < n; ++e) {
        auto edge = lookup_table.get_edge(node, e);
        core::action_t action = edge->action;
        // Game::Symmetries::apply(action, inv_sym, node->action_mode());
        actions(e) = action;
        A2(e) = edge->A;
        AVs_original(e) = AV_original(e, seat);
        AVs(e) = edge->child_AV[seat];
        AUs(e) = edge->child_AU[seat];
        lAVs(e) = edge->child_lAUV[seat].mean();
        lAUs(e) = edge->child_lAUV[seat].variance();

        LogitValueArray child_lAUV;
        Calculations<Game>::populate_logit_value_beliefs(AV_original.row(e), AU.row(e), child_lAUV);
        lAVs_original(e) = child_lAUV[seat].mean();
      }

      static std::vector<std::string> action_columns = {"action", "AV_orig", "AV", "AU", "lAV_orig",
                                                        "lAV",    "lAU",     "P",  "A"};

      auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
        actions, AVs_original, AVs, AUs, lAVs_original, lAVs, lAUs, P, A2));

      eigen_util::PrintArrayFormatMap fmt_map2{
        {"action", [&](float x) { return Game::IO::action_to_str(x, node->action_mode()); }},
      };

      std::stringstream ss2;
      eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

      for (const std::string& line : util::splitlines(ss2.str())) {
        ss << line << line_break;
      }

      LOG_INFO(ss.str());
    }
  }

  if (root) {
    root->stats().N = std::max(root->stats().N, 1);
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_results(const GeneralContext& general_context, SearchResults& results) {
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;

  const Node* root = lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here

  core::seat_index_t seat = stable_data.active_seat;
  core::action_mode_t mode = root->action_mode();

  results.valid_actions.reset();
  results.P.setZero();
  results.pi.setZero();
  results.AQ.setZero();
  results.AU.setZero();

  core::action_t actions[stable_data.num_valid_actions];

  bool provably_lost = stats.Q[seat] == Game::GameResults::kMinValue;

  int i = 0;
  for (core::action_t action : stable_data.valid_action_mask.on_indices()) {
    results.valid_actions.set(action, true);
    actions[i] = action;

    const Edge* edge = lookup_table.get_edge(root, i);
    const Node* child = lookup_table.get_node(edge->child_index);

    results.P(action) = edge->P;

    if (provably_lost) {
      // if losing, just play according to prior
      results.pi(action) = edge->P;
    } else {
      results.pi(action) = edge->pi;
    }

    const auto& AQ = child ? child->stats().Q : edge->child_AV;
    const auto& AU = child ? child->stable_data().U : edge->child_AU;
    const auto& AV = child ? child->stable_data().V() : edge->child_AV;
    const auto& AW = child ? child->stats().W : edge->child_AU;

    for (int p = 0; p < kNumPlayers; ++p) {
      results.AQ(action, p) = AQ[p];
      results.AU(action, p) = AU[p];
      results.AV(action, p) = AV[p];
      results.AW(action, p) = AW[p];
    }
    i++;
  }

  RELEASE_ASSERT(eigen_util::any(results.pi));
  eigen_util::normalize(results.pi);
  results.R = stable_data.R;
  results.Q = stats.Q;
  results.Q_min = stats.Q_min;
  results.Q_max = stats.Q_max;
  results.W = stats.W;
  results.seat = stable_data.active_seat;

  x0::Algorithms<Traits>::load_action_symmetries(general_context, root, &actions[0], results);
  results.action_mode = mode;
  results.provably_lost = provably_lost;

  if (search::kEnableSearchDebug) {
    std::ostringstream ss;
    std::string line_break = std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength);

    ss << "SEARCH RESULTS" << line_break;

    ValueArray players;
    ValueArray V = stable_data.V();
    ValueArray CP;
    for (int p = 0; p < kNumPlayers; ++p) {
      players(p) = p;
      CP(p) = p == seat;
    }

    static std::vector<std::string> player_columns = {"Seat",  "V", "Q",   "Q_min",
                                                      "Q_max", "W", "CurP"};
    auto player_data = eigen_util::concatenate_columns(players, V, results.Q, results.Q_min,
                                                       results.Q_max, results.W, CP);

    eigen_util::PrintArrayFormatMap fmt_map1{
      {"Seat", [&](float x) { return std::to_string(int(x)); }},
      {"CurP", [&](float x) { return std::string(x ? "*" : ""); }},
    };

    std::stringstream ss1;
    eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

    for (const std::string& line : util::splitlines(ss1.str())) {
      ss << line << line_break;
    }

    LocalPolicyArray action_array(stable_data.num_valid_actions);
    LocalPolicyArray pi_array(stable_data.num_valid_actions);
    LocalPolicyArray AQ_array(stable_data.num_valid_actions);
    LocalPolicyArray AU_array(stable_data.num_valid_actions);

    for (int e = 0; e < stable_data.num_valid_actions; ++e) {
      core::action_t action = actions[e];

      action_array(e) = action;
      pi_array(e) = results.pi(action);
      AQ_array(e) = results.AQ(action, seat);
      AU_array(e) = results.AU(action, seat);
    }

    static std::vector<std::string> action_columns = {"action", "pi", "AQ", "AU"};
    auto action_data = eigen_util::sort_rows(
      eigen_util::concatenate_columns(action_array, pi_array, AQ_array, AU_array));

    eigen_util::PrintArrayFormatMap fmt_map{
      {"action", [&](float x) { return Game::IO::action_to_str(x, mode); }},
    };

    std::stringstream ss2;
    eigen_util::print_array(ss2, action_data, action_columns, &fmt_map);

    for (const std::string& line : util::splitlines(ss2.str())) {
      ss << line << line_break;
    }

    LOG_INFO(ss.str());
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::write_to_training_info(const TrainingInfoParams& params,
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
    training_info.policy_target = mcts_results->pi;
    training_info.policy_target_valid =
      x0::Algorithms<Traits>::validate_and_symmetrize_policy_target(mcts_results,
                                                                    training_info.policy_target);
  }
  if (use_for_training) {
    training_info.action_values_target = mcts_results->AQ;
    training_info.action_values_target_valid = true;
  }

  training_info.Q = eigen_util::reinterpret_as_tensor(mcts_results->Q);
  training_info.Q_min = eigen_util::reinterpret_as_tensor(mcts_results->Q_min);
  training_info.Q_max = eigen_util::reinterpret_as_tensor(mcts_results->Q_max);
  training_info.W = eigen_util::reinterpret_as_tensor(mcts_results->W);

  if (params.use_for_training) {
    training_info.AU = mcts_results->AU;
    training_info.AU_valid = true;
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_record(const TrainingInfo& training_info,
                                   GameLogFullRecord& full_record) {
  alpha0::Algorithms<Traits>::to_record(training_info, full_record);

  full_record.Q = training_info.Q;
  full_record.Q_min = training_info.Q_min;
  full_record.Q_max = training_info.Q_max;
  full_record.W = training_info.W;

  if (training_info.AU_valid) {
    full_record.AU = training_info.AU;
  } else {
    full_record.AU.setZero();
  }
  full_record.AU_valid = training_info.AU_valid;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::serialize_record(const GameLogFullRecord& full_record,
                                          std::vector<char>& buf) {
  GameLogCompactRecord compact_record;
  compact_record.position = full_record.position;
  compact_record.Q = full_record.Q;
  compact_record.Q_min = full_record.Q_min;
  compact_record.Q_max = full_record.Q_max;
  compact_record.W = full_record.W;
  compact_record.active_seat = full_record.active_seat;
  compact_record.action_mode = Game::Rules::get_action_mode(full_record.position);
  compact_record.action = full_record.action;

  PolicyTensorData policy(full_record.policy_target_valid, full_record.policy_target);
  ActionValueTensorData action_values(full_record.action_values_valid, full_record.action_values);
  ActionValueTensorData AU(full_record.AU_valid, full_record.AU);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values.write_to(buf);
  AU.write_to(buf);
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_view(const GameLogViewParams& params, GameLogView& view) {
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
  const PolicyTensorData* policy_data = reinterpret_cast<const PolicyTensorData*>(policy_data_addr);

  const char* action_values_data_addr = policy_data_addr + policy_data->size();
  const ActionValueTensorData* action_values_data =
    reinterpret_cast<const ActionValueTensorData*>(action_values_data_addr);

  const char* AU_data_addr = action_values_data_addr + action_values_data->size();
  const ActionValueTensorData* AU_data =
    reinterpret_cast<const ActionValueTensorData*>(AU_data_addr);

  view.policy_valid = policy_data->load(view.policy);
  view.action_values_valid = action_values_data->load(view.action_values);
  view.AU_valid = AU_data->load(view.AU);

  if (view.policy_valid) {
    Game::Symmetries::apply(view.policy, sym, mode);
  }

  if (view.action_values_valid) {
    Game::Symmetries::apply(view.action_values, sym, mode);
  }

  if (view.AU_valid) {
    Game::Symmetries::apply(view.AU, sym, mode);
  }

  view.next_policy_valid = false;
  if (next_record) {
    const char* next_addr = reinterpret_cast<const char*>(next_record);

    const char* next_policy_data_addr = next_addr + sizeof(GameLogCompactRecord);
    const PolicyTensorData* next_policy_data =
      reinterpret_cast<const PolicyTensorData*>(next_policy_data_addr);

    view.next_policy_valid = next_policy_data->load(view.next_policy);
    if (view.next_policy_valid) {
      Game::Symmetries::apply(view.next_policy, sym, next_record->action_mode);
    }
  }

  view.cur_pos = *cur_pos;
  view.final_pos = *final_pos;
  view.game_result = *outcome;
  view.active_seat = active_seat;
  view.Q = record->Q;
  view.Q_min = record->Q_min;
  view.Q_max = record->Q_max;
  view.W = record->W;
}

}  // namespace beta0
