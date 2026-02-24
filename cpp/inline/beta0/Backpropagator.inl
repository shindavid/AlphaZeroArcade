#include "beta0/Backpropagator.hpp"

#include "beta0/Constants.hpp"
#include "search/Constants.hpp"
#include "util/EigenUtil.hpp"
#include "util/Gaussian1D.hpp"

#include <sstream>

namespace beta0 {

template <search::concepts::Traits Traits>
template <typename MutexProtectedFunc>
Backpropagator<Traits>::Backpropagator(SearchContext& context, Node* node, Edge* edge,
                                       MutexProtectedFunc&& func)
    : context_(context), node_(node) {
  RELEASE_ASSERT(!node_->stable_data().is_chance_node, "Chance nodes not yet supported");

  preload_parent_data();
  load_parent_data(func);
  load_remaining_data();
  compute_update_rules();
  apply_updates();
  if (search::kEnableSearchDebug) print_debug_info();
}

template <search::concepts::Traits Traits>
bool Backpropagator<Traits>::shares_mutex_with_parent(const Node* child) const {
  return &child->mutex() == &node_->mutex();
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::load_child_stats(int k, const NodeStats& child_stats) {
  read_data_(r_lQ, k) = child_stats.lQW[seat_].mean();
  read_data_(r_lW, k) = child_stats.lQW[seat_].variance();
  read_data_(r_Q, k) = child_stats.Q[seat_];
  read_data_(r_W, k) = child_stats.W[seat_];
  read_data_(r_R, k) = child_stats.R;
  read_data_(r_child_N, k) = child_stats.N;
  RELEASE_ASSERT(read_data_(r_lW, k) != util::Gaussian1D::kVarianceUnset, "Invalid lW value");
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::preload_parent_data() {
  // Only read data that doesn't require locking the parent mutex.

  const auto& stable_data = node_->stable_data();
  RELEASE_ASSERT(!stable_data.is_chance_node, "Chance nodes not yet supported");

  n_ = stable_data.num_valid_actions;
  seat_ = stable_data.active_seat;

  read_data_.resize(n_);

  for (int k = 0; k < n_; k++) {
    const Edge* child_edge = lookup_table().get_edge(node_, k);
    const Node* child = lookup_table().get_node(child_edge->child_index);

    read_data_(r_E, k) = child != nullptr ? 1.f : 0.f;
    read_data_(r_AV, k) = child_edge->child_AV[seat_];
    if (child) {
      const auto& lUV_k = child->stable_data().lUV[seat_];
      read_data_(r_lV, k) = lUV_k.mean();
      read_data_(r_lU, k) = lUV_k.variance();
      RELEASE_ASSERT(lUV_k.variance() != util::Gaussian1D::kVarianceUnset, "Invalid lU value");
    } else {
      const auto& lUV_k = child_edge->child_lAUV[seat_];
      read_data_(r_lV, k) = lUV_k.mean();
      read_data_(r_lU, k) = lUV_k.variance();
      read_data_(r_lQ, k) = read_data_(r_lV, k);
      read_data_(r_lW, k) = read_data_(r_lU, k);
      read_data_(r_R, k) = 0.f;
      read_data_(r_child_N, k) = 0.f;

      read_data_(r_Q, k) = child_edge->child_AV[seat_];
      read_data_(r_W, k) = child_edge->child_AU[seat_];
      RELEASE_ASSERT(lUV_k.variance() != util::Gaussian1D::kVarianceUnset, "Invalid lU value2");
      RELEASE_ASSERT(read_data_(r_lW, k) != util::Gaussian1D::kVarianceUnset, "Invalid lW value2");
    }
  }
}

template <search::concepts::Traits Traits>
template <typename MutexProtectedFunc>
void Backpropagator<Traits>::load_parent_data(MutexProtectedFunc&& func) {
  // Now read data that requires locking the parent mutex.

  mit::unique_lock lock(node_->mutex());

  func();
  stats_ = node_->stats();  // copy

  for (int k = 0; k < n_; k++) {
    const Edge* child_edge = lookup_table().get_edge(node_, k);
    read_data_(r_P, k) = child_edge->P;
    read_data_(r_pi, k) = child_edge->pi;
    read_data_(r_A, k) = child_edge->A;
    read_data_(r_A_neg_inf, k) = child_edge->pi == 0.f;
    read_data_(r_edge_N, k) = child_edge->child_N;
    read_data_(r_prev_lQ, k) = child_edge->previous_lQW.mean();
    read_data_(r_prev_lW, k) = child_edge->previous_lQW.variance();

    if (read_data_(r_E, k)) {
      const Node* child = lookup_table().get_node(child_edge->child_index);
      if (shares_mutex_with_parent(child)) {
        // child shares same mutex as parent, so we can read its stats directly
        load_child_stats(k, child->stats());
      } else {
        // locking child mutex can cause deadlock, so defer loading child stats until later
        deferred_child_stats_load_indices_[num_deferred_child_stats_load_indices_++] = k;
      }
    }
  }
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::load_remaining_data() {
  // 0. Read children data that we deferred earlier to avoid deadlocks.

  for (int c = 0; c < num_deferred_child_stats_load_indices_; c++) {
    int k = deferred_child_stats_load_indices_[c];
    const Edge* child_edge = lookup_table().get_edge(node_, k);
    const Node* child = lookup_table().get_node(child_edge->child_index);

    const auto child_stats = child->stats_safe();  // make a copy - acquires child mutex
    load_child_stats(k, child_stats);
  }

  const auto E = read_data_(r_E);
  const auto Q = read_data_(r_Q);
  const auto W = read_data_(r_W);
  const auto child_N = read_data_(r_child_N);
  const auto edge_N = read_data_(r_edge_N);

  // 0. Compute fresh_indicies_

  for (int k = 0; k < n_; ++k) {
    fresh_indices_.set(k, child_N(k) != edge_N(k));
  }

  // 1. Compute E_mask_

  E_mask_ = Mask::Zero(n_);
  E_mask_ = E > 0.f;
  U_mask_ = !E_mask_;

  // 2. Compute Q_floor_

  Q_floor_ = Game::GameResults::kMinValue;
  Mask W0_mask = Mask::Zero(n_);
  W0_mask = E_mask_ && (W == 0.f);

  int W0_count = W0_mask.count();
  if (W0_count) {
    Q_floor_ = eigen_util::mask_splice(Q, W0_mask).maxCoeff();
  }
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_update_rules() {
  full_write_data_.resize(n_);
  update_Q_estimates();
  compute_ratings();
  compute_policy();
  update_R();
  update_QW();

  stats_.Q_min = stats_.Q_min.cwiseMin(stats_.Q);
  stats_.Q_max = stats_.Q_max.cwiseMax(stats_.Q);

  // TODO: remove below check
  if (!(read_data_(r_lW) == util::Gaussian1D::kVariancePosInf).any()) {
    for (int k = 0; k < n_; k++) {
      if (read_data_(r_A, k) == 0.f) {
        bool fail = read_data_(r_lW, k) != util::Gaussian1D::kVarianceNegInf;
        if (fail) {
          print_debug_info();
          RELEASE_ASSERT(false, "Inconsistent A and lW values for action index {}", k);
        }
        fail = read_data_(r_pi, k) != 0.f;
        if (fail) {
          print_debug_info();
          RELEASE_ASSERT(false, "Inconsistent A and pi values for action index {}", k);
        }
      }
    }
  }
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::apply_updates() {
  mit::unique_lock lock(node_->mutex());

  for (int k = 0; k < n_; k++) {
    Edge* child_edge = lookup_table().get_edge(node_, k);
    child_edge->pi = full_write_data_(fw_pi, k);
    child_edge->A = full_write_data_(fw_A, k);
    child_edge->child_N = read_data_(r_child_N, k);
    child_edge->previous_lQW = util::Gaussian1D(read_data_(r_lQ, k), read_data_(r_lW, k));
  }

  int N = node_->stats().N;
  node_->stats() = stats_;  // copy back
  node_->stats().N = N;
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::print_debug_info() {
  std::ostringstream ss;
  ss << std::format("{:>{}}", "", context_.log_prefix_n());

  ValueArray players;
  ValueArray nQ = stats_.Q;
  ValueArray nW = stats_.W;
  ValueArray n_lQ;
  ValueArray n_lW;
  ValueArray beta;
  ValueArray CP;

  for (int p = 0; p < kNumPlayers; ++p) {
    players(p) = p;
    CP(p) = p == seat_;
    n_lQ(p) = stats_.lQW[p].mean();
    n_lW(p) = stats_.lQW[p].variance();
    beta(p) = node_->stable_data().beta0;
  }

  static std::vector<std::string> player_columns = {"Seat", "Q", "W", "lQ", "lW", "beta0", "CurP"};
  auto player_data = eigen_util::concatenate_columns(players, nQ, nW, n_lQ, n_lW, beta, CP);

  eigen_util::PrintArrayFormatMap fmt_map_a{
    {"Seat", [&](float x) { return std::to_string(int(x)); }},
    {"CurP", [&](float x) { return std::string(x ? "*" : ""); }},
  };

  std::stringstream ss_a;
  eigen_util::print_array(ss_a, player_data, player_columns, &fmt_map_a);

  std::string line_break =
    std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context_.log_prefix_n());

  for (const std::string& line : util::splitlines(ss_a.str())) {
    ss << line << line_break;
  }
  ss << line_break;

  const LocalArray E = read_data_(r_E);
  const LocalArray R = read_data_(r_R);
  const LocalArray P = read_data_(r_P);
  const LocalArray pi_before = read_data_(r_pi);
  const LocalArray A_before = read_data_(r_A);
  const LocalArray lV = read_data_(r_lV);
  const LocalArray lU = read_data_(r_lU);
  const LocalArray lQ_before = read_data_(r_lQ);
  const LocalArray lW = read_data_(r_lW);
  const LocalArray Q = read_data_(r_Q);
  const LocalArray W = read_data_(r_W);
  const LocalArray AV = read_data_(r_AV);

  const LocalArray Q_capped_after = full_write_data_(fw_Q);
  const LocalArray lQ_after = full_write_data_(fw_lQ);
  const LocalArray pi_after = full_write_data_(fw_pi);
  const LocalArray A_after = full_write_data_(fw_A);

  LocalArray actions(n_);
  LocalArray i_indicator(n_);
  LocalArray N(n_);

  const auto& search_path = context_.search_path;
  for (const auto& visitation : search_path) {
    if (visitation.node == node_) {
      break;
    }
  }

  for (int e = 0; e < n_; ++e) {
    auto edge = lookup_table().get_edge(node_, e);
    actions(e) = edge->action;
    i_indicator(e) = fresh_indices_[e];

    auto child = lookup_table().get_node(edge->child_index);
    N(e) = child ? child->stats().N : 0;
  }

  static std::vector<std::string> action_columns = {"action", "i",  "E",   "N",   "R",  "P", "pi",
                                                    "A",      "lV", "lU",  "lQ",  "lW", "Q", "AV",
                                                    "W",      "Q*", "lQ*", "pi*", "A*"};

  auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
    actions, i_indicator, E, N, R, P, pi_before, A_before, lV, lU, lQ_before, lW, Q, AV, W,
    Q_capped_after, lQ_after, pi_after, A_after));

  eigen_util::PrintArrayFormatMap fmt_map_b1{
    {"action", [&](float x) { return Game::IO::action_to_str(x, node_->action_mode()); }},
    {"i", [](float x) { return x == 0.f ? "" : "*"; }},
    {"lU", util::Gaussian1D::fmt_variance},
    {"lW", util::Gaussian1D::fmt_variance},
  };

  eigen_util::PrintArrayFormatMap2 fmt_map_b2{
    {"lV", {"lU", util::Gaussian1D::fmt_mean}},
    {"lQ", {"lW", util::Gaussian1D::fmt_mean}},
    {"lQ*", {"lW", util::Gaussian1D::fmt_mean}},
  };

  std::stringstream ss_b;
  eigen_util::print_array(ss_b, action_data, action_columns, &fmt_map_b1, &fmt_map_b2);

  for (const std::string& line : util::splitlines(ss_b.str())) {
    ss << line << line_break;
  }

  LOG_INFO(ss.str());
}

template <search::concepts::Traits Traits>
bool Backpropagator<Traits>::handle_edge_cases() {
  int pos_inf_count = 0;
  int neg_inf_count = 0;
  int zero_lW_count = 0;
  float max_zero_lW_value = std::numeric_limits<float>::lowest();

  for (int k = 0; k < n_; k++) {
    float lW_k = read_data_(r_lW, k);
    bool pos_inf = (lW_k == util::Gaussian1D::kVariancePosInf);
    bool neg_inf = (lW_k == util::Gaussian1D::kVarianceNegInf);
    bool zero = (lW_k == 0.f);

    pos_inf_count += pos_inf;
    neg_inf_count += neg_inf;
    zero_lW_count += zero;

    if (pos_inf_count) break;
    if (neg_inf) {
      full_write_data_(fw_A, k) = 0.f;  // we set to 0 by convention
      full_write_data_(fw_A_neg_inf, k) = 1.f;
    }
    if (zero) {
      float lQ_k = read_data_(r_lQ, k);
      if (lQ_k > max_zero_lW_value) {
        max_zero_lW_value = lQ_k;
      }
    }
  }

  if (pos_inf_count) {
    for (int k = 0; k < n_; k++) {
      if (read_data_(r_lW, k) == util::Gaussian1D::kVariancePosInf) {
        full_write_data_(fw_A, k) = 1.f;  // arbitrary value
        full_write_data_(fw_A_neg_inf, k) = 0.f;
      } else {
        full_write_data_(fw_A, k) = 0.f;  // we set to 0 by convention
        full_write_data_(fw_A_neg_inf, k) = 1.f;
      }
    }
    return true;
  }

  if (neg_inf_count == n_) {
    // All actions have -inf rating. This is a losing position.
    full_write_data_(fw_A).setZero();
    full_write_data_(fw_A_neg_inf).setConstant(1.f);
    return true;
  }

  if (neg_inf_count + 1 == n_) {
    // All but one action have -inf rating. Put all policy mass on the remaining action.
    for (int k = 0; k < n_; k++) {
      if (read_data_(r_lW, k) != util::Gaussian1D::kVarianceNegInf) {
        full_write_data_(fw_A, k) = 1.f;  // arbitrary value
        full_write_data_(fw_A_neg_inf, k) = 0.f;
        break;
      }
    }
    return true;
  }

  // If we got here, there are no +inf actions, and at least two finite actions.

  if (zero_lW_count) {
    // Zero out zero-variance actions dominated by other zero-variance actions
    for (int k = 0; k < n_; k++) {
      float lW_k = read_data_(r_lW, k);
      float lQ_k = read_data_(r_lQ, k);
      if (lW_k == 0.f && lQ_k < max_zero_lW_value) {
        full_write_data_(fw_A, k) = 0.f;  // we set to 0 by convention
        full_write_data_(fw_A_neg_inf, k) = 1.f;
      }
    }
    if (zero_lW_count + neg_inf_count == n_) {
      // There are no uncertain actions
      return true;
    }
  }

  return false;
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::update_Q_estimates() {
  auto AV = read_data_(r_AV);
  auto Q = read_data_(r_Q);
  auto W = read_data_(r_W);
  auto E = read_data_(r_E);
  auto P = read_data_(r_P);
  auto R = read_data_(r_R);
  float beta0 = node_->stable_data().beta0;

  constexpr float kMin = Game::GameResults::kMinValue;
  constexpr float kMax = Game::GameResults::kMaxValue;

  Mask min_mask = Mask::Zero(n_);
  min_mask = Q == kMin;

  Mask max_mask = Mask::Zero(n_);
  max_mask = Q == Game::GameResults::kMaxValue;

  auto YY = E * P;
  auto XX = YY * (Q - AV);
  float XXs = XX.sum();
  float YYs = YY.sum();

  LocalArray X = XXs - XX;
  LocalArray Y = YYs - YY;

  eigen_util::reassign(Y, 0.f, 1.f);
  LocalArray XY = X / Y;
  XY = XY.cwiseMax(kMin - kMax).cwiseMin(kMax - kMin);
  auto gain = eigen_util::logit(0.5f * (1 + kSiblingGain * XY));
  auto beta = beta0 + gain;
  auto beta_factor = W.sqrt() * (R + 1).rsqrt();

  LocalArray Q_out = eigen_util::sigmoid(eigen_util::logit(Q) + beta_factor * beta);
  Q_out = min_mask.select(Game::GameResults::kMinValue, Q_out);
  Q_out = max_mask.select(Game::GameResults::kMaxValue, Q_out);
  LocalArray lQ_out(Q_out.size());
  Calculations::p2l(Q_out, W, lQ_out);

  full_write_data_(fw_Q) = Q_out;
  full_write_data_(fw_lQ) = lQ_out;
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_ratings() {
  full_write_data_(fw_A) = read_data_(r_A);
  full_write_data_(fw_A_neg_inf) = read_data_(r_A_neg_inf);
  if (!handle_edge_cases()) {
    for (int i : fresh_indices_.on_indices()) {
      if (compute_ratings_helper(i)) break;
    }
  }
}

template <search::concepts::Traits Traits>
bool Backpropagator<Traits>::compute_ratings_helper(int i) {
  const util::Gaussian1D lQW_i(full_write_data_(fw_lQ, i), read_data_(r_lW, i));
  if (lQW_i == util::Gaussian1D::neg_inf()) {
    safety_check(__LINE__);
    return false;
  }
  if (read_data_(r_pi, i) >= 1.f) {
    safety_check(__LINE__);
    return true;  // all policy mass is already on this action
  }

  RELEASE_ASSERT(lQW_i.valid());

  const float P_i = read_data_(r_P, i);
  const float lQ_i = full_write_data_(fw_lQ, i);
  const float lW_i = read_data_(r_lW, i);
  const float lV_i = read_data_(r_lV, i);
  const float lU_i = read_data_(r_lU, i);

  const float lQ_old_i = read_data_(r_prev_lQ, i);
  const float lW_old_i = read_data_(r_prev_lW, i);

  const auto P = splice(read_data_(r_P), i);
  const auto lV = splice(read_data_(r_lV), i);
  const auto lU = splice(read_data_(r_lU), i);
  const auto lW = splice(read_data_(r_lW), i);

  const auto lQ = splice(full_write_data_(fw_lQ), i);

  const int n = n_ - 1;
  auto lU_rsqrt = (lU_i + lU).rsqrt();
  LocalArray c = (lV_i - lV) * lU_rsqrt;
  auto P_i_ratio = P_i / (P_i + P);
  LocalArray z = kInvBeta * eigen_util::logit(P_i_ratio) - c;

  LocalArray tau = compute_tau(lQ_i, lQ, lW_i, lW, z, lU_rsqrt);
  if (tau.isConstant(1.0f, 0.0f)) {
    // all tau are 1 - put all policy mass on this action
    full_write_data_(fw_A).fill(0.f);  // we set to 0 by convention
    full_write_data_(fw_A, i) = 1.f;  // arbitrary value
    full_write_data_(fw_A_neg_inf).fill(1.f);
    full_write_data_(fw_A_neg_inf, i) = 0.f;
    safety_check(__LINE__);
    return true;
  } else if (tau.isZero(0.f)) {
    // all tau are 0 - put no policy mass on this action
    full_write_data_(fw_A, i) = 0.f;  // we set to 0 by convention
    full_write_data_(fw_A_neg_inf, i) = 1.f;
    safety_check(__LINE__);
    return false;
  }

  LocalArray tau_old = compute_tau(lQ_old_i, lQ, lW_old_i, lW, z, lU_rsqrt);

  LocalArray tau_opp = 1.f - tau;
  LocalArray tau_opp_old = 1.f - tau_old;

  // sanity check: if tau_opp_old is 0, then tau_opp is also 0:
  Mask tau_opp_old_zero_mask = Mask::Zero(n);
  tau_opp_old_zero_mask = tau_opp_old == 0.f;
  auto masked_tau_opp = eigen_util::mask_splice(tau_opp, tau_opp_old_zero_mask);
  if (!masked_tau_opp.isZero(0.f)) {
    print_debug_info();
    RELEASE_ASSERT(false, "Inconsistent tau and tau_old values");
  }

  // if tau_opp_old is 0, then tau_opp must also be 0, justifying this reassign:
  eigen_util::reassign(tau_opp_old, 0.f, 1.f);
  LocalArray tau_opp_ratio = tau_opp * eigen_util::invert(tau_opp_old);

  LocalArray A_adj = kGamma * P_i * eigen_util::invert(1.f - P) * tau_opp_ratio.log();

  auto A = full_write_data_(fw_A);
  auto A_neg_inf = full_write_data_(fw_A_neg_inf);
  A += unsplice(A_adj, i);
  A_neg_inf = (A_neg_inf > 0 || unsplice(tau_opp, i) == 0.f).template cast<float>();

  float A_i = 0.f;
  float A_neg_inf_i = (tau == 0.f).any();

  if (!A_neg_inf_i) {
    float log_P_i = std::log(P_i);
    float A_i_num = (P * (tau.log() - P_i_ratio.log())).sum();
    float A_i_den = 1 - P_i;
    A_i = log_P_i + kGamma * A_i_num / A_i_den;
  }

  full_write_data_(fw_A, i) = A_i;
  full_write_data_(fw_A_neg_inf, i) = A_neg_inf_i;
  safety_check(__LINE__);
  return false;
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_policy() {
  // pi obtained by softmaxing the nonzero A values

  auto A = full_write_data_(fw_A);
  auto A_neg_inf = full_write_data_(fw_A_neg_inf);
  auto pi = full_write_data_(fw_pi);

  Mask finite_A_mask = A_neg_inf == 0.f;

  if (!finite_A_mask.any()) {
    pi.setConstant(1.f / n_);
    return;
  }
  LocalArray pi_nonzero = eigen_util::mask_splice(A, finite_A_mask);
  eigen_util::softmax_in_place(pi_nonzero);

  LocalArray pi_out = pi * 0.f;
  eigen_util::mask_splice_assign(pi, finite_A_mask, pi_nonzero);
  pi = pi_out;
}

template <search::concepts::Traits Traits>
typename Backpropagator<Traits>::LocalArray Backpropagator<Traits>::compute_tau(
  float lQ_i, const LocalArray& lQ, float lW_i, const LocalArray& lW, const LocalArray& z,
  const LocalArray& lU_rsqrt) {
  // Effectively computes:
  //
  // sigmoid(kBeta [ (lQ_i - lQ) / (lW_i + lW).sqrt() + z * sqrt((lW_i + lW) / (lU_i + lU)) ])
  //
  // But does some special casing for infinities and zero variances. Each (lQ, lW) pair comes
  // from a util::Gaussian1D, so we need to handle the various edge cases corresponding to how
  // util::Gaussian1D represents +/- infinity.

  if (lW_i == util::Gaussian1D::kVarianceNegInf) {
    return lQ * 0;
  } else if (lW_i == util::Gaussian1D::kVariancePosInf) {
    return lQ * 0 + 1.0f;
  } else {
    int n = z.size();

    Mask mask = Mask::Zero(n);
    mask = lW >= 0.f;
    if (lW_i == 0.f) {
      mask = mask && ((lW > 0.f) || (lQ == lQ_i));
    }
    int mn = mask.count();
    LocalArray lQ_m = eigen_util::mask_splice(lQ, mask);
    LocalArray lW_m = eigen_util::mask_splice(lW, mask);
    LocalArray z_m = eigen_util::mask_splice(z, mask);
    LocalArray lU_rsqrt_m = eigen_util::mask_splice(lU_rsqrt, mask);

    LocalArray num = (lQ_i - lQ_m) + (lW_i + lW_m) * z_m * lU_rsqrt_m;
    LocalArray inv_den = (lW_i + lW_m).rsqrt();
    LocalArray tau_m = eigen_util::sigmoid(kBeta * (num * inv_den));
    tau_m = tau_m.cwiseMax(1e-6f).cwiseMin(1.0f - 1e-6f);  // clamp for stability

    if (mn == n) {
      return tau_m;
    }

    LocalArray tau(n);
    int m = 0;
    for (int j = 0; j < n; j++) {
      if (mask[j]) {
        tau[j] = tau_m[m++];
        continue;
      }

      // If we get here, then either lW[j] < 0 (i.e., +inf or -inf), or lW_i == lW[j] == 0 with
      // lQ_i != lQ(j).

      float lW_j = lW(j);
      bool tau1 = lW_j == util::Gaussian1D::kVarianceNegInf || (lW_j == 0.f && lQ_i > lQ(j));
      tau[j] = tau1 ? 1.0f : 0.0f;
    }

    return tau;
  }
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::update_R() {
  auto R = read_data_(r_R);
  auto pi = full_write_data_(fw_pi);

  float pi_max = pi.maxCoeff();
  stats_.R = 1.0f + (R * pi).sum() / pi_max;
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::update_QW() {
  // Q/W-Update rules:
  //
  // Q_c = sum_i pi_i * Q^*_i
  // W_c = sum_i pi_i * (W^*_i + (Q^*_i - Q_c)^2))
  // R_c = R(p) - 1
  //
  // Q(p) = (W_c * V(p) + R_c * U(p) * Q_c) / (W_c + R_c * U(p))
  // W(p) = (W_c * U(p) * (1 + R_c)) / (W_c + R_c * U(p))
  //
  // where Q^*_i is the *conditional* belief
  //
  // Q^*_i = E[Z_i | i = argmax_j Z_j]
  //
  // and W^*_i is the corresponding conditional uncertainty.
  //
  // We approximate Q^*_i = max(Q_i, Q_floor), where
  //
  // Q_floor is the maximum Q_k over all actions k with W_k = 0 (i.e. no uncertainty).
  //
  // And we approximate W^*_i = W_i.

  auto pi = full_write_data_(fw_pi);
  auto W = read_data_(r_W);
  auto Q = full_write_data_(fw_Q);

  auto Q_capped = Q.cwiseMax(Q_floor_);

  float Q_c = Calculations::exact_dot_product(Q_capped, pi);

  auto W_across = (Q_capped - Q_c).square();
  float W_c = Calculations::exact_dot_product(W + W_across, pi);
  float R_c = stats_.R - 1.0f;

  float Up = node_->stable_data().U[seat_];
  float Vp = node_->stable_data().V()[seat_];

  float RU = R_c * Up;
  float denom = W_c + RU;

  float Qp_num = W_c * Vp + RU * Q_c;
  float Wp_num = W_c * Up * (1.0f + R_c);

  float Qp = Qp_num / denom;
  float Wp = Wp_num / denom;

  stats_.Q[seat_] = Qp;
  stats_.W[seat_] = Wp;

  if (kNumPlayers == 2) {
    // In two-player zero-sum games, the opponent's Q/W are just the negative of the player's.
    stats_.Q[1-seat_] = -Qp;
    stats_.W[1-seat_] = Wp;
  }
}

// TODO: remove this check once confident
template <search::concepts::Traits Traits>
void Backpropagator<Traits>::safety_check(int line) {
  bool fail = (full_write_data_(fw_A) == 0.f).all();
  if (fail) {
    print_debug_info();
  }
  RELEASE_ASSERT(!fail, "All A values are zero - cannot proceed (line {})", line);
}

template <search::concepts::Traits Traits>
typename Backpropagator<Traits>::LocalArray Backpropagator<Traits>::splice(const LocalArray& x,
                                                                           int i) {
  LocalArray to(n_ - 1);
  if (i > 0) {
    to.topRows(i) = x.topRows(i);
  }

  int tail = n_ - i - 1;
  if (tail > 0) {
    to.bottomRows(tail) = x.bottomRows(tail);
  }

  return to;
}

template <search::concepts::Traits Traits>
typename Backpropagator<Traits>::LocalArray Backpropagator<Traits>::unsplice(const LocalArray& x,
                                                                             int i) {
  LocalArray result(n_);
  const auto from = x;
  if (i > 0) {
    result.topRows(i) = from.topRows(i);
  }

  result[i] = 0.f;  // unused

  int tail = n_ - i - 1;
  if (tail > 0) {
    result.bottomRows(tail) = from.bottomRows(tail);
  }
  return result;
}

}  // namespace beta0
