#include "beta0/Algorithms.hpp"

#include "alpha0/Algorithms.hpp"
#include "beta0/Constants.hpp"
#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "util/EigenUtil.hpp"
#include "util/Gaussian1D.hpp"
#include "util/LoggingUtil.hpp"
#include "util/MetaProgramming.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep
#include "x0/Algorithms.hpp"

#include <cmath>

namespace beta0 {

template <search::concepts::Traits Traits>
template <typename MutexProtectedFunc>
void Algorithms<Traits>::backprop(SearchContext& context, Node* node, Edge* edge,
                                  MutexProtectedFunc&& func) {
  mit::unique_lock lock(node->mutex());
  func();
  if (!edge) return;
  NodeStats stats = node->stats();  // copy
  lock.unlock();

  update_stats(stats, node, context);

  lock.lock();

  // Carefully copy back fields of stats back to node->stats()
  // We don't copy counts, which may have been updated by other threads.
  int N = node->stats().N;
  node->stats() = stats;
  node->stats().N = N;
  lock.unlock();
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::init_node_stats_from_terminal(Node* node) {
  const ValueArray q = node->stable_data().V();

  NodeStats& stats = node->stats();
  RELEASE_ASSERT(stats.N == 0);
  RELEASE_ASSERT(stats.R == 0.f);

  stats.Q = q;
  stats.Q_min = stats.Q;
  stats.Q_max = stats.Q;
  stats.W.fill(0.f);
  Calculations::p2l(stats.Q, stats.W, stats.lQW);
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::update_node_stats(Node* node, bool undo_virtual) {
  node->stats().N++;
  node->stats().R += 1.f;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::update_node_stats_and_edge(Node* node, Edge* edge, bool undo_virtual) {
  edge->E++;
  node->stats().N++;
  node->stats().R += 1.f;
}

template <search::concepts::Traits Traits>
bool Algorithms<Traits>::should_short_circuit(const Edge* edge, const Node* child) {
  int edge_count = edge->E;
  int child_count = child->stats().N;  // not thread-safe but race-condition is benign
  return edge_count < child_count;
}

template <search::concepts::Traits Traits>
bool Algorithms<Traits>::more_search_iterations_needed(const GeneralContext& general_context,
                                                       const Node* root) {
  // root->stats() usage here is not thread-safe but this race-condition is benign
  const search::SearchParams& search_params = general_context.search_params;
  if (!search_params.ponder && root->stable_data().num_valid_moves == 1) return false;
  if (root->stats().W.isZero(0.f)) return false;
  if (root->stats().move_forced) return false;
  return root->stats().N <= search_params.tree_size_limit;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::init_root_info(GeneralContext& general_context,
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
  const State& cur_state = root_info.state;
  if (root_info.node_index < 0) {
    root_info.node_index = lookup_table.alloc_node();
    Node* root = lookup_table.get_node(root_info.node_index);

    core::seat_index_t active_seat = Game::Rules::get_current_player(cur_state);
    RELEASE_ASSERT(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info.active_seat = active_seat;
    auto legal_moves = Game::Rules::analyze(cur_state).valid_moves();
    new (root) Node(lookup_table.get_random_mutex(), cur_state, legal_moves.size(), active_seat);
  }

  if (search::kEnableSearchDebug && purpose == search::kForStandardSearch) {
    Game::IO::print_state(std::cout, cur_state);
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::init_root_edges(GeneralContext& general_context) {}

template <search::concepts::Traits Traits>
int Algorithms<Traits>::get_best_child_index(const SearchContext& context) {
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
    int other_seat = 1 - seat;

    int n = stable_data.num_valid_moves;

    GameResultTensor R;
    ValueArray U01;
    LocalActionValueArray AU01(n, kNumPlayers);
    LocalPolicyArray P_raw(n);
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

    std::copy_n(eval->data(0), P_raw.size(), P_raw.data());
    std::copy_n(eval->data(1), R.size(), R.data());
    std::copy_n(eval->data(2), AV.size(), AV.data());
    std::copy_n(eval->data(3), U01.size(), U01.data());
    std::copy_n(eval->data(4), AU01.size(), AU01.data());

    RELEASE_ASSERT(eigen_util::isfinite(P_raw), "Non-finite P");
    RELEASE_ASSERT(eigen_util::isfinite(R), "Non-finite R");
    RELEASE_ASSERT(eigen_util::isfinite(U01), "Non-finite U01");
    RELEASE_ASSERT(eigen_util::isfinite(AV), "Non-finite AV");
    RELEASE_ASSERT(eigen_util::isfinite(AU01), "Non-finite AU01");

    LocalPolicyArray P_adjusted = P_raw;
    transform_policy(context, P_adjusted);

    Array2D AV_original = AV;

    ValueArray V = GameResultEncoding::to_value_array(R);

    // clamp to avoid extreme values
    constexpr float kMin = GameResultEncoding::kMinValue + 1e-6f;
    constexpr float kMax = GameResultEncoding::kMaxValue - 1e-6f;
    V = V.cwiseMax(kMin).cwiseMin(kMax);
    AV = AV.cwiseMax(kMin).cwiseMin(kMax);
    AU01 = AU01.cwiseMax(1e-6f);  // avoid 0 uncertainty

    ValueArray U = Calculations::scale_uncertainty(V, U01);
    Array2D AU = Calculations::scale_uncertainty(AV, AU01);

    Array2D AU_original = AU;

    Array2D lAV = Array2D::Zero(n, kNumPlayers);
    Array2D lAU = Array2D::Zero(n, kNumPlayers);
    Calculations::p2l(AV, AU, lAV, lAU);

    ValueArray U_original = U;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wrestrict"
    Array2D lAV_original = lAV;
#pragma GCC diagnostic pop

    Array1D lAVs = lAV.col(seat);
    Array1D lAUs = lAU.col(seat);

    int lAVs_max_index;
    lAVs.maxCoeff(&lAVs_max_index);
    float lAVs_max = lAVs[lAVs_max_index];
    float lAUs_at_max = lAUs[lAVs_max_index];

    Array1D z = (lAVs - lAVs_max) * (lAUs_at_max + lAUs).rsqrt();
    Array1D tau = eigen_util::sigmoid(kBeta * z);
    Array1D pi = tau / tau.sum();

    RELEASE_ASSERT(z[lAVs_max_index] == 0.f, "z[{}]={} (!= 0.0f)", lAVs_max_index,
                   z[lAVs_max_index]);
    RELEASE_ASSERT(tau[lAVs_max_index] == 0.5f, "tau[{}] = {} (!= 0.5f)", lAVs_max_index,
                   tau[lAVs_max_index]);

    float beta = Calculations::compute_beta(V[seat], pi, lAVs);

    lAV.col(seat) += beta;
    if (kNumPlayers == 2) {
      lAV.col(other_seat) = -lAV.col(seat);
    }

    Calculations::l2p(lAV, lAU, AV, AU);

    AV = AV.cwiseMax(kMin).cwiseMin(kMax);
    AU = AU.cwiseMax(1e-9f);

    Array1D AUs = AU.col(seat);
    float Vs = V[seat];
    Array1D AVs = AV.col(seat);

    Array1D Ucontrib_s = AUs + (Vs - AVs) * (Vs - AVs);
    U[seat] = Calculations::exact_dot_product(Ucontrib_s, pi);
    if (kNumPlayers == 2) {
      U[other_seat] = U[seat];
    }

    for (int i = 0; i < n; ++i) {
      Edge* edge = lookup_table.get_edge(node, i);
      edge->P_raw = P_raw[i];
      edge->P_adjusted = P_adjusted[i];
      edge->child_AV = AV.row(i);
      edge->child_AU = AU.row(i);
      edge->lVUs = util::Gaussian1D(lAV(i, seat), lAU(i, seat));
    }

    stable_data.R = R;
    stable_data.R_valid = true;
    stable_data.U = U;

    stats.Q = V;
    stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
    stats.Q_max = stats.Q_max.cwiseMax(stats.Q);
    stats.W = U;
    Calculations::p2l(stats.Q, stats.W, stats.lQW);

    if (search::kEnableSearchDebug) {
      std::ostringstream ss;

      ss << "NN EVAL\n\n";
      ss << "beta:  " << beta << "\n";

      ValueArray players;
      ValueArray CP;
      ValueArray lV;
      ValueArray lU;

      for (int p = 0; p < kNumPlayers; ++p) {
        players(p) = p;
        CP(p) = p == seat;
        lV(p) = stats.lQW[p].mean();
        lU(p) = stats.lQW[p].variance();
      }

      static std::vector<std::string> player_columns = {"Seat", "V", "U", "Uo", "lV", "lU", "CurP"};
      auto player_data = eigen_util::concatenate_columns(players, V, U, U_original, lV, lU, CP);

      eigen_util::PrintArrayFormatMap fmt_map1{
        {"Seat", [&](float x, int) { return std::to_string(int(x)); }},
        {"CurP", [&](float x, int) { return std::string(x ? "*" : ""); }},
      };

      eigen_util::print_array(ss, player_data, player_columns, &fmt_map1);
      ss << "\n";

      Array1D AVos(n);
      Array1D AUos(n);
      Array1D lAVos(n);
      Array1D lAVs2 = lAV.col(seat);
      Array1D AU01s = AU01.col(seat);

      for (int e = 0; e < n; ++e) {
        AVos(e) = AV_original(e, seat);
        AUos(e) = AU_original(e, seat);
        lAVos(e) = lAV_original(e, seat);
      }

      MoveSet valid_moves = lookup_table.get_moves(node);
      ActionPrinter printer(valid_moves);
      Array1D actions = printer.flat_array();

      static std::vector<std::string> action_columns = {"action", "AVo",  "AV",  "AU01", "AUo",
                                                        "AU",     "lAVo", "lAV", "lAU",  "Pr",
                                                        "Pa",     "z",    "tau", "pi"};

      auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
        actions, AVos, AVs, AU01s, AUos, AUs, lAVos, lAVs2, lAUs, P_raw, P_adjusted, z, tau, pi));

      eigen_util::PrintArrayFormatMap fmt_map2;
      printer.update_format_map(fmt_map2);

      eigen_util::print_array(ss, action_data, action_columns, &fmt_map2);
      util::Logging::multi_line_log_info(ss.str(), context.log_prefix_n());
    }

    float V2 = Calculations::exact_dot_product(AVs, pi);
    RELEASE_ASSERT(std::abs(Vs - V2) < .01f, "compute_beta() failed to converge ({} vs {})", Vs,
                   V2);
  }

  if (root) {
    root->stats().N = std::max(root->stats().N, 1);
    root->stats().R = std::max(root->stats().R, 1.0f);
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_results(const GeneralContext& general_context, SearchResults& results) {
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  const Node* root = lookup_table.get_node(root_info.node_index);
  const auto& stable_data = root->stable_data();
  const auto& stats = root->stats();  // thread-safe since single-threaded here

  core::seat_index_t seat = stable_data.active_seat;
  core::game_phase_t game_phase = root->game_phase();

  results.frame = root_info.input_encoder.current_frame();
  results.valid_moves.clear();
  results.P.setZero();
  results.pre_expanded_moves.setZero();

  results.AV.setZero();
  results.AU.setZero();
  results.AQ.setZero();
  results.AQ_min.setZero();
  results.AQ_max.setZero();
  results.AW.setZero();
  results.N.setZero();
  results.RN.setZero();
  results.policy_target.setZero();

  PolicyTensor PW;
  PolicyTensor PL;

  PW.setZero();
  PL.setZero();

  bool provably_lost = stats.Q[seat] == GameResultEncoding::kMinValue;
  bool provably_won = stats.Q[seat] == GameResultEncoding::kMaxValue;

  int n = stable_data.num_valid_moves;
  int i = 0;
  auto valid_moves = Game::Rules::analyze(root_info.state).valid_moves();
  results.valid_moves = valid_moves;
  for (Move move : valid_moves) {
    auto index = PolicyEncoding::to_index(results.frame, move);

    const Edge* edge = lookup_table.get_edge(root, i);
    const Node* child = lookup_table.get_node(edge->child_index);

    results.P.coeffRef(index) = edge->P_raw;
    results.pre_expanded_moves.coeffRef(index) = edge->was_pre_expanded;
    results.N.coeffRef(index) = child ? child->stats().N : 0;
    results.RN.coeffRef(index) = child ? child->stats().R : 0.f;
    results.policy_target.coeffRef(index) = (n == 1) ? 1.0f : edge->E;

    const auto& AV = child ? child->stable_data().V() : edge->child_AV;
    const auto& AU = child ? child->stable_data().U : edge->child_AU;
    const auto& AQ = child ? child->stats().Q : edge->child_AV;
    const auto& Q_min = child ? child->stats().Q_min : edge->child_AV;
    const auto& Q_max = child ? child->stats().Q_max : edge->child_AV;
    const auto& AW = child ? child->stats().W : edge->child_AU;

    for (int p = 0; p < kNumPlayers; ++p) {
      auto index_p = eigen_util::extend_index(index, p);
      results.AV.coeffRef(index_p) = AV[p];
      results.AU.coeffRef(index_p) = AU[p];
      results.AQ.coeffRef(index_p) = AQ[p];
      results.AQ_min.coeffRef(index_p) = Q_min[p];
      results.AQ_max.coeffRef(index_p) = Q_max[p];
      results.AW.coeffRef(index_p) = AW[p];
    }

    PW.coeffRef(index) = AW[seat] == 0.0f && AQ[seat] == GameResultEncoding::kMaxValue;
    PL.coeffRef(index) = AW[seat] == 0.0f && AQ[seat] == GameResultEncoding::kMinValue;

    i++;
  }

  RELEASE_ASSERT(eigen_util::isfinite(results.AQ), "Non-finite AQ");

  results.policy = results.policy_target;
  if (!provably_lost) {
    results.policy *= (1 - PL);
  }
  if (provably_won) {
    results.policy *= PW;
  }
  eigen_util::normalize(results.policy);
  eigen_util::normalize(results.policy_target);

  results.R = stable_data.R;
  results.Q = stats.Q;
  results.Q_min = stats.Q_min;
  results.Q_max = stats.Q_max;
  results.W = stats.W;
  results.seat = stable_data.active_seat;

  x0::Algorithms<Traits>::load_action_symmetries(general_context, root, results);
  results.game_phase = game_phase;
  results.provably_lost = provably_lost;

  if (manager_params.forced_playouts && root_info.add_noise) {
    prune_policy_target(general_context, results);
  }

  if (search::kEnableSearchDebug) {
    std::ostringstream ss;
    ss << "SEARCH RESULTS\n";

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
      {"Seat", [&](float x, int) { return std::to_string(int(x)); }},
      {"CurP", [&](float x, int) { return std::string(x ? "*" : ""); }},
    };

    eigen_util::print_array(ss, player_data, player_columns, &fmt_map1);

    LocalPolicyArray P_array(n);
    LocalPolicyArray E_array(n);
    LocalPolicyArray AV_array(n);
    LocalPolicyArray AU_array(n);
    LocalPolicyArray AW_array(n);
    LocalPolicyArray AQ_array(n);
    LocalPolicyArray AQ_min_array(n);
    LocalPolicyArray AQ_max_array(n);
    LocalPolicyArray policy_array(n);

    ActionPrinter printer(valid_moves);
    LocalPolicyArray action_array = printer.flat_array();

    int e = 0;
    for (Move move : valid_moves) {
      auto index = PolicyEncoding::to_index(results.frame, move);
      auto index_s = eigen_util::extend_index(index, seat);

      P_array(e) = results.policy_target.coeff(index);
      E_array(e) = results.N.coeff(index);
      AV_array(e) = results.AV.coeff(index_s);
      AU_array(e) = results.AU.coeff(index_s);
      AW_array(e) = results.AW.coeff(index_s);
      AQ_array(e) = results.AQ.coeff(index_s);
      AQ_min_array(e) = results.AQ_min.coeff(index_s);
      AQ_max_array(e) = results.AQ_max.coeff(index_s);
      policy_array(e) = results.policy.coeff(index);

      e++;
    }

    static std::vector<std::string> action_columns = {"action", "P",  "E",      "AV",     "AU",
                                                      "AW",     "AQ", "AQ_min", "AQ_max", "pi"};
    auto action_data = eigen_util::sort_rows(
      eigen_util::concatenate_columns(action_array, P_array, E_array, AV_array, AU_array, AW_array,
                                      AQ_array, AQ_min_array, AQ_max_array, policy_array));

    eigen_util::PrintArrayFormatMap fmt_map;
    printer.update_format_map(fmt_map);

    eigen_util::print_array(ss, action_data, action_columns, &fmt_map);
    util::Logging::multi_line_log_info(ss.str());
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::write_to_training_info(const TrainingInfoParams& params,
                                                TrainingInfo& training_info) {
  const SearchResults* mcts_results = params.mcts_results;

  bool use_for_training = params.use_for_training;
  bool previous_used_for_training = params.previous_used_for_training;
  core::seat_index_t seat = params.seat;

  training_info.frame = params.frame;
  training_info.active_seat = seat;
  training_info.move = params.move;
  training_info.use_for_training = use_for_training;

  if (use_for_training || previous_used_for_training) {
    training_info.policy_target = mcts_results->policy_target;
    training_info.policy_target_valid =
      x0::Algorithms<Traits>::validate_and_symmetrize_policy_target(mcts_results,
                                                                    training_info.policy_target);
  }

  if (use_for_training) {
    training_info.action_values_target =
      x0::Algorithms<Traits>::apply_mask(mcts_results->AQ, mcts_results->pre_expanded_moves);
    training_info.action_values_target_valid = true;
  }

  training_info.Q = eigen_util::reinterpret_as_tensor(mcts_results->Q);
  training_info.Q_min = eigen_util::reinterpret_as_tensor(mcts_results->Q_min);
  training_info.Q_max = eigen_util::reinterpret_as_tensor(mcts_results->Q_max);
  training_info.W = eigen_util::reinterpret_as_tensor(mcts_results->W);

  if (use_for_training) {
    training_info.AQ_min =
      x0::Algorithms<Traits>::apply_mask(mcts_results->AQ_min, mcts_results->pre_expanded_moves);
    training_info.AQ_max =
      x0::Algorithms<Traits>::apply_mask(mcts_results->AQ_max, mcts_results->pre_expanded_moves);
    training_info.AU =
      x0::Algorithms<Traits>::apply_mask(mcts_results->AU, mcts_results->pre_expanded_moves);
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

  if (training_info.action_values_target_valid) {
    full_record.AQ_min = training_info.AQ_min;
    full_record.AQ_max = training_info.AQ_max;
    full_record.AU = training_info.AU;
  } else {
    full_record.AQ_min.setZero();
    full_record.AQ_max.setZero();
    full_record.AU.setZero();
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::serialize_record(const GameLogFullRecord& full_record,
                                          std::vector<char>& buf) {
  GameLogCompactRecord compact_record;
  compact_record.frame = full_record.frame;
  compact_record.Q = full_record.Q;
  compact_record.Q_min = full_record.Q_min;
  compact_record.Q_max = full_record.Q_max;
  compact_record.W = full_record.W;
  compact_record.active_seat = full_record.active_seat;
  compact_record.game_phase = full_record.game_phase;
  compact_record.move = full_record.move;

  PolicyTensorData policy(full_record.policy_target_valid, full_record.policy_target);
  ActionValueTensorData action_values(full_record.action_values_valid, full_record.action_values);
  ActionValueTensorData AQ_min(full_record.action_values_valid, full_record.AQ_min);
  ActionValueTensorData AQ_max(full_record.action_values_valid, full_record.AQ_max);
  ActionValueTensorData AU(full_record.action_values_valid, full_record.AU);

  search::GameLogCommon::write_section(buf, &compact_record, 1, false);
  policy.write_to(buf);
  action_values.write_to(buf);
  AQ_min.write_to(buf);
  AQ_max.write_to(buf);
  AU.write_to(buf);
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::to_view(const GameLogViewParams& params, GameLogView& view) {
  const GameLogCompactRecord* record = params.record;
  const GameLogCompactRecord* next_record = params.next_record;
  const InputFrame* cur_frame = params.cur_frame;
  const InputFrame* final_frame = params.final_frame;
  const GameResultTensor* outcome = params.outcome;
  group::element_t sym = params.sym;

  core::seat_index_t active_seat = record->active_seat;
  core::game_phase_t game_phase = record->game_phase;

  const char* addr = reinterpret_cast<const char*>(record);

  const char* policy_data_addr = addr + sizeof(GameLogCompactRecord);
  const PolicyTensorData* policy_data = reinterpret_cast<const PolicyTensorData*>(policy_data_addr);

  const char* action_values_data_addr = policy_data_addr + policy_data->size();
  const ActionValueTensorData* action_values_data =
    reinterpret_cast<const ActionValueTensorData*>(action_values_data_addr);

  const char* AQ_min_data_addr = action_values_data_addr + action_values_data->size();
  const ActionValueTensorData* AQ_min_data =
    reinterpret_cast<const ActionValueTensorData*>(AQ_min_data_addr);

  const char* AQ_max_data_addr = AQ_min_data_addr + AQ_min_data->size();
  const ActionValueTensorData* AQ_max_data =
    reinterpret_cast<const ActionValueTensorData*>(AQ_max_data_addr);

  const char* AU_data_addr = AQ_max_data_addr + AQ_max_data->size();
  const ActionValueTensorData* AU_data =
    reinterpret_cast<const ActionValueTensorData*>(AU_data_addr);

  view.policy_valid = policy_data->load(view.policy);
  view.action_values_valid = action_values_data->load(view.action_values);
  view.action_values_valid &= AQ_min_data->load(view.AQ_min);
  view.action_values_valid &= AQ_max_data->load(view.AQ_max);
  view.action_values_valid &= AU_data->load(view.AU);

  if (view.policy_valid) {
    Symmetries::apply(view.policy, sym, game_phase);
  }

  if (view.action_values_valid) {
    Symmetries::apply(view.action_values, sym, game_phase);
    Symmetries::apply(view.AQ_min, sym, game_phase);
    Symmetries::apply(view.AQ_max, sym, game_phase);
    Symmetries::apply(view.AU, sym, game_phase);
  }

  view.next_policy_valid = false;
  if (next_record) {
    const char* next_addr = reinterpret_cast<const char*>(next_record);

    const char* next_policy_data_addr = next_addr + sizeof(GameLogCompactRecord);
    const PolicyTensorData* next_policy_data =
      reinterpret_cast<const PolicyTensorData*>(next_policy_data_addr);

    view.next_policy_valid = next_policy_data->load(view.next_policy);
    if (view.next_policy_valid) {
      Symmetries::apply(view.next_policy, sym, next_record->game_phase);
    }
  }

  view.cur_frame = *cur_frame;
  view.final_frame = *final_frame;
  view.game_result = *outcome;
  view.active_seat = active_seat;
  view.Q = record->Q;
  view.Q_min = record->Q_min;
  view.Q_max = record->Q_max;
  view.W = record->W;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::update_stats(NodeStats& stats, const Node* node, SearchContext& context) {
  LookupTable& lookup_table = context.general_context->lookup_table;
  const auto& stable_data = node->stable_data();
  int n = stable_data.num_valid_moves;
  core::seat_index_t seat = stable_data.active_seat;

  if (stable_data.is_chance_node) {
    throw util::Exception("Chance nodes not yet supported in beta0");
  }

  Array1D AV = Array1D::Zero(n);
  Array1D Q = Array1D::Zero(n);
  Array1D W = Array1D::Zero(n);
  Array1D lQ = Array1D::Zero(n);
  Array1D lW = Array1D::Zero(n);
  Array1D P = Array1D::Zero(n);
  Array1D N = Array1D::Zero(n);
  Array1D E = Array1D::Zero(n);
  Array1D S = Array1D::Zero(n);
  Array1D pi = Array1D::Zero(n);
  Array1D tau = Array1D::Zero(n);

  RELEASE_ASSERT(n > 0);
  for (int i = 0; i < n; i++) {
    const Edge* edge = lookup_table.get_edge(node, i);
    const Node* child = lookup_table.get_node(edge->child_index);

    P(i) = edge->P_raw;
    AV(i) = edge->child_AV[seat];
    if (!child) {
      Q(i) = edge->child_AV[seat];
      W(i) = edge->child_AU[seat];
      lQ(i) = edge->lVUs.mean();
      lW(i) = edge->lVUs.variance();
    } else {
      const auto child_stats = child->stats_safe();  // make a copy
      Q(i) = child_stats.Q[seat];
      W(i) = child_stats.W[seat];
      lQ(i) = child_stats.lQW[seat].mean();
      lW(i) = child_stats.lQW[seat].variance();
      E(i) = 1.0f;
      N(i) = child_stats.N;
    }
  }

  int argmax_Q;
  Q.maxCoeff(&argmax_Q);

  Array1D Q_star = Q;
  bool move_forced = false;

  float Qp, Wp;
  float Q_max = Q(argmax_Q);
  float W_max = W.maxCoeff();
  if (W_max == 0.0f) {
    // Every child has a deterministically known result. Use the one with the max Q value.
    pi[argmax_Q] = 1.0f;  // just for printing
    Qp = Q_max;
    Wp = 0.0f;
    move_forced = true;
  } else if (Q_max == GameResultEncoding::kMaxValue) {
    // Provably winning.
    pi[argmax_Q] = 1.0f;  // just for printing
    Qp = Q_max;
    Wp = 0.0f;
    move_forced = true;
  } else if (Q_max == GameResultEncoding::kMinValue) {
    // Provably losing.
    Qp = Q_max;
    Wp = 0.0f;
    move_forced = true;
  } else {
    tau.setConstant(-1.0f);  // initialize to invalid value for temp debugging
    float lW_at_max = lW(argmax_Q);
    float lQ_max = lQ(argmax_Q);

    // First fill in tau where the normals overlap
    Array1D lW_sum = lW + lW_at_max;
    Mask finite_mask = lW >= 0.f;
    Mask tau_mask = finite_mask && (lW_sum > 0.f);

    Array1D lW_sum_t = eigen_util::mask_splice(lW_sum, tau_mask);
    Array1D lQ_diff_t = eigen_util::mask_splice(lQ - lQ_max, tau_mask);
    Array1D z_t = lQ_diff_t * lW_sum_t.rsqrt();
    Array1D tau_t = eigen_util::sigmoid(kBeta * z_t);
    eigen_util::mask_splice_assign(tau, tau_mask, tau_t);

    // Now fill in edge-cases of tau
    Mask neg_inf_mask = Q == GameResultEncoding::kMinValue;
    Mask pos_inf_mask = Q == GameResultEncoding::kMaxValue;
    Mask finite_zero_W_mask = (lW == 0.f) && finite_mask;
    Mask finite_zero_W_mask_eq = finite_zero_W_mask && (Q == Q_max);
    Mask finite_zero_W_mask_lt = finite_zero_W_mask && (Q < Q_max);

    tau = neg_inf_mask.select(0.0f, tau);
    tau = pos_inf_mask.select(1.0f, tau);
    tau = finite_zero_W_mask_eq.select(0.5f, tau);
    tau = finite_zero_W_mask_lt.select(0.0f, tau);

    pi = tau / tau.sum();

    // Compute surprise vector S
    Array1D S_den_base = E * P;
    Array1D S_num_base = S_den_base * (Q - AV);
    Array1D S_num = S_num_base.sum() - S_num_base;
    Array1D S_den = S_den_base.sum() - S_den_base;
    S_den = S_den.cwiseMax(1e-6f);  // avoid divide-by-zero
    Array1D S_den_inv = eigen_util::invert(S_den);
    S = S_num * S_den_inv;

    // Compute Q_star by shifting Q in the logit space based on S. We need to be careful to only
    // compute it where W > 0 to avoid logit of 0 or 1.
    Mask mask = W > 0.0f;

    Array1D Sm = eigen_util::mask_splice(S, mask);
    Array1D Nm = eigen_util::mask_splice(N, mask);
    Array1D lQm = eigen_util::mask_splice(lQ, mask);

    // TODO: consider using R instead of N here
    Array1D Q_star_m = eigen_util::sigmoid(lQm + kSurpriseGain * Sm * (Nm + 1).rsqrt());

    eigen_util::mask_splice_assign(Q_star, mask, Q_star_m);

    // TODO: consider capping Q_star by Q_floor, the max Q among children with W == 0

    Qp = Calculations::exact_dot_product(Q_star, pi);
    Array1D W_across = (Q_star - Qp).square();
    Wp = Calculations::exact_dot_product(W + W_across, pi);

    // if W=0 and Q=GameResultEncoding::kMinValue for all siblings of argmax_Q, then our move is
    // forced

    auto kMin = GameResultEncoding::kMinValue;
    auto kMax = GameResultEncoding::kMaxValue;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    Mask losing_mask = (Q == kMin) && (W == 0.0f);
    if (losing_mask.count() == n - 1) {
      move_forced = true;
    }
#pragma GCC diagnostic pop

    Qp = std::clamp(Qp, kMin + 1e-6f, kMax - 1e-6f);
    Wp = std::max(Wp, 1e-10f);
  }

  stats.Q[seat] = Qp;
  stats.W[seat] = Wp;
  stats.move_forced = move_forced;
  if (kNumPlayers == 2) {
    stats.Q[1 - seat] = 1.0f - Qp;
    stats.W[1 - seat] = Wp;
  }
  Calculations::p2l(stats.Q, stats.W, stats.lQW);

  stats.Q_min = stats.Q_min.cwiseMin(stats.Q);
  stats.Q_max = stats.Q_max.cwiseMax(stats.Q);

  if (!search::kEnableSearchDebug) return;

  std::ostringstream ss;

  ss << "update_stats()\n\n";

  ValueArray players;
  ValueArray CP;
  for (int p = 0; p < kNumPlayers; ++p) {
    players(p) = p;
    CP(p) = p == seat;
  }

  static std::vector<std::string> player_columns = {"Seat", "Q", "W", "CurP"};
  auto player_data = eigen_util::concatenate_columns(players, stats.Q, stats.W, CP);

  eigen_util::PrintArrayFormatMap fmt_map1{
    {"Seat", [&](float x, int) { return std::to_string(int(x)); }},
    {"CurP", [&](float x, int) { return std::string(x ? "*" : ""); }},
  };

  eigen_util::print_array(ss, player_data, player_columns, &fmt_map1);
  ss << "\n";

  MoveSet valid_moves = lookup_table.get_moves(node);
  ActionPrinter printer(valid_moves);
  Array1D actions = printer.flat_array();
  Array1D maxQ(n);

  for (int e = 0; e < n; ++e) {
    maxQ(e) = argmax_Q == e ? 1.0f : 0.0f;
  }

  static std::vector<std::string> action_columns = {"action", "E", "maxQ", "Q", "W",  "lQ",  "lW",
                                                    "AV",     "N", "P",    "S", "Q*", "tau", "pi"};

  auto action_data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions, E, maxQ, Q, W, lQ, lW, AV, N, P, S, Q_star, tau, pi));

  eigen_util::PrintArrayFormatMap fmt_map2{
    {"E", [&](float x, int) { return std::string(x ? "*" : ""); }},
    {"maxQ", [&](float x, int) { return std::string(x ? "*" : ""); }},
    {"lW", [](float x, int) { return util::Gaussian1D::fmt_variance(x); }},
    {"lQ", [&](float x, int row) { return util::Gaussian1D::fmt_mean(x, lW[row]); }},
  };
  printer.update_format_map(fmt_map2);

  eigen_util::print_array(ss, action_data, action_columns, &fmt_map2);
  util::Logging::multi_line_log_info(ss.str(), context.log_prefix_n());
}

template <search::concepts::Traits Traits>
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
      float temp = general_context.aux_state.root_softmax_temperature.value();
      if (temp > 0.0f) {
        P = P.pow(1.0f / temp);
      }
      P /= P.sum();
    }
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::add_dirichlet_noise(GeneralContext& general_context, LocalPolicyArray& P) {
  const ManagerParams& manager_params = general_context.manager_params;
  auto& dirichlet_gen = general_context.aux_state.dirichlet_gen;
  auto& rng = general_context.aux_state.rng;

  int n = P.rows();
  double alpha = manager_params.dirichlet_alpha_factor / sqrt(n);
  LocalPolicyArray noise = dirichlet_gen.template generate<LocalPolicyArray>(rng, alpha, n);
  P = (1.0 - manager_params.dirichlet_mult) * P + manager_params.dirichlet_mult * noise;
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::prune_policy_target(const GeneralContext& general_context,
                                             SearchResults& results) {
  const search::SearchParams& search_params = general_context.search_params;
  const RootInfo& root_info = general_context.root_info;
  const LookupTable& lookup_table = general_context.lookup_table;
  const ManagerParams& manager_params = general_context.manager_params;

  if (manager_params.no_model) return;

  const Node* root = lookup_table.get_node(root_info.node_index);
  PuctCalculator action_selector(lookup_table, manager_params, search_params, root, true);

  const auto& P = action_selector.P;
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

  int n_actions = root->stable_data().num_valid_moves;
  for (int i = 0; i < n_actions; ++i) {
    const Edge* edge = lookup_table.get_edge(root, i);
    const Move& move = edge->move;
    auto index = PolicyEncoding::to_index(results.frame, move);

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
    results.policy_target = results.N;
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::print_action_selection_details(const SearchContext& context,
                                                        const PuctCalculator& selector,
                                                        int argmax_index) {
  if (!search::kEnableSearchDebug) return;

  LookupTable& lookup_table = context.general_context->lookup_table;
  Node* node = context.visit_node;

  std::ostringstream ss;

  core::seat_index_t seat = node->stable_data().active_seat;

  int n_actions = node->stable_data().num_valid_moves;

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
  const LocalPolicyArray& W = selector.W;
  const LocalPolicyArray& E = selector.E;
  const LocalPolicyArray& mE = selector.mE;
  const LocalPolicyArray& N = selector.N;
  const LocalPolicyArray& PUCT = selector.PUCT;

  LocalPolicyArray child_addr(n_actions);
  LocalPolicyArray argmax(n_actions);
  child_addr.setConstant(-1);
  argmax.setZero();
  argmax(argmax_index) = 1;

  for (int e = 0; e < n_actions; ++e) {
    auto edge = lookup_table.get_edge(node, e);
    child_addr(e) = edge->child_index;
  }

  MoveSet valid_moves = lookup_table.get_moves(node);
  ActionPrinter printer(valid_moves);
  Array1D actions = printer.flat_array();

  static std::vector<std::string> action_columns = {"action", "P", "Q",   "W",    "E",
                                                    "mE",     "N", "&ch", "PUCT", "argmax"};
  auto action_data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions, P, Q, W, E, mE, N, child_addr, PUCT, argmax));

  eigen_util::PrintArrayFormatMap fmt_map2{
    {"&ch", [](float x, int) { return x < 0 ? std::string() : std::to_string((int)x); }},
    {"argmax", [](float x, int) { return std::string(x == 0 ? "" : "*"); }},
  };
  printer.update_format_map(fmt_map2);

  eigen_util::print_array(ss, action_data, action_columns, &fmt_map2);
  util::Logging::multi_line_log_info(ss.str(), context.log_prefix_n());
}

}  // namespace beta0
