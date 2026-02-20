#include "beta0/Backpropagator.hpp"

#include "beta0/Constants.hpp"
#include "search/Constants.hpp"
#include "util/EigenUtil.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"

#include <sstream>

namespace beta0 {

template <search::concepts::Traits Traits>
template <typename MutexProtectedFunc>
Backpropagator<Traits>::Backpropagator(SearchContext& context, Node* node, Edge* edge,
                                       MutexProtectedFunc&& func)
    : context_(context), node_(node), edge_(edge) {
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

  int i = -1;
  for (int k = 0; k < n_; k++) {
    const Edge* child_edge = lookup_table().get_edge(node_, k);
    const Node* child = lookup_table().get_node(child_edge->child_index);

    if (child_edge == edge_) {
      i = k;
      child_i_ = child;
    }

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

      // TODO: for non-expanded children, I don't think r_Q and r_W are used. Does that mean we
      // can remove Edge::child_AV and Edge::child_AU?
      read_data_(r_Q, k) = child_edge->child_AV[seat_];
      read_data_(r_W, k) = child_edge->child_AU[seat_];
      RELEASE_ASSERT(lUV_k.variance() != util::Gaussian1D::kVarianceUnset, "Invalid lU value2");
      RELEASE_ASSERT(read_data_(r_lW, k) != util::Gaussian1D::kVarianceUnset, "Invalid lW value2");
    }
  }

  RELEASE_ASSERT(i >= 0, "Edge not found in parent node");
  i_ = i;
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

  // 1 Compute E_mask_

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

  // 3. Copy read data into sibling_read_data_

  sibling_read_data_.resize(n_ - 1);

  r_splice(r_P, sr_P);
  r_splice(r_pi, sr_pi);
  r_splice(r_lV, sr_lV);
  r_splice(r_lU, sr_lU);
  r_splice(r_lW, sr_lW);
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_update_rules() {
  full_write_data_.resize(n_);
  sibling_write_data_.resize(n_ - 1);
  update_Q_estimates();
  compute_ratings();
  calibrate_ratings();
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
    beta(p) = stats_.beta;
  }

  static std::vector<std::string> player_columns = {"Seat", "Q", "W", "lQ", "lW", "beta", "CurP"};
  auto player_data = eigen_util::concatenate_columns(players, nQ, nW, n_lQ, n_lW, beta, CP);

  eigen_util::PrintArrayFormatMap fmt_map1{
    {"Seat", [&](float x) { return std::to_string(int(x)); }},
    {"CurP", [&](float x) { return std::string(x ? "*" : ""); }},
  };

  std::stringstream ss1;
  eigen_util::print_array(ss1, player_data, player_columns, &fmt_map1);

  std::string line_break =
    std::format("\n{:>{}}", "", util::Logging::kTimestampPrefixLength + context_.log_prefix_n());

  for (const std::string& line : util::splitlines(ss1.str())) {
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

  const LocalArray c = unsplice(sw_c);
  const LocalArray z = unsplice(sw_z);
  const LocalArray w = unsplice(sw_w);
  const LocalArray tau = unsplice(sw_tau);

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
    i_indicator(e) = (e == i_) ? 1.f : 0.f;

    auto child = lookup_table().get_node(edge->child_index);
    N(e) = child ? child->stats().N : 0;
  }

  static std::vector<std::string> action_columns = {
    "action", "i", "E",  "N", "R", "P", "pi", "A",   "lV",  "lU",  "lQ", "lW",
    "Q",      "AV", "W", "Q*", "c", "z", "w",  "tau", "lQ*", "pi*", "A*"};

  auto action_data = eigen_util::sort_rows(eigen_util::concatenate_columns(
    actions, i_indicator, E, N, R, P, pi_before, A_before, lV, lU, lQ_before, lW, Q, AV, W,
    Q_capped_after, c, z, w, tau, lQ_after, pi_after, A_after));

  eigen_util::PrintArrayFormatMap fmt_map2{
    {"action", [&](float x) { return Game::IO::action_to_str(x, node_->action_mode()); }},
    {"i", [](float x) { return x == 0.f ? "" : "*"; }},
  };

  std::stringstream ss2;
  eigen_util::print_array(ss2, action_data, action_columns, &fmt_map2);

  for (const std::string& line : util::splitlines(ss2.str())) {
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
      full_write_data_(fw_A, k) = 0.f;  // 0 means -inf
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
        full_write_data_(fw_A, k) = -1.f;
      } else {
        full_write_data_(fw_A, k) = 0.f;  // 0 means -inf
      }
    }
    return true;
  }

  if (neg_inf_count == n_) {
    // All actions have -inf rating. This is a losing position. Arbitrarily set an action to -1.
    full_write_data_(fw_A, i_) = -1.f;
    return true;
  }

  if (neg_inf_count + 1 == n_) {
    // All but one action have -inf rating. Put all policy mass on the remaining action.
    for (int k = 0; k < n_; k++) {
      if (read_data_(r_lW, k) != util::Gaussian1D::kVarianceNegInf) {
        full_write_data_(fw_A, k) = -1.f;
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
        full_write_data_(fw_A, k) = 0.f;  // 0 means -inf
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
  float beta = stats_.beta;

  auto Q_out = full_write_data_(fw_Q);
  auto lQ_out = full_write_data_(fw_lQ);

  auto YY = E * P;
  auto XX = YY * (Q - AV);
  float XXs = XX.sum();
  float YYs = YY.sum();

  LocalArray X = XXs - XX;
  LocalArray Y = YYs - YY;

  // wherever Y is 0, set it to 1 to avoid NaNs:
  Y = Y.unaryExpr([](float y) { return y == 0.f ? 1.f : y; });

  auto gain = eigen_util::logit(0.5f * (1 + kSiblingGain * X / Y));
  Q_out = beta + gain;
  LocalArray lQ(X.size());
  Calculations::p2l(Q_out, W, lQ);
  lQ_out = lQ;
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_ratings() {
  const util::Gaussian1D lQW_i(full_write_data_(fw_lQ, i_), read_data_(r_lW, i_));
  full_write_data_(fw_A) = read_data_(r_A);
  if (handle_edge_cases()) {
    safety_check(__LINE__);
    return;
  }

  if (lQW_i == util::Gaussian1D::neg_inf()) {
    safety_check(__LINE__);
    return;
  }
  if (read_data_(r_pi, i_) >= 1.f) {
    safety_check(__LINE__);
    return;  // all policy mass is already on this action
  }

  w_splice(fw_A, sw_A);
  w_splice(fw_lQ, sw_lQ);

  RELEASE_ASSERT(lQW_i.valid());

  const float P_i = read_data_(r_P, i_);
  float A_i = read_data_(r_A, i_);
  const float lQ_i = full_write_data_(fw_lQ, i_);
  const float lW_i = read_data_(r_lW, i_);
  const float lV_i = read_data_(r_lV, i_);
  const float lU_i = read_data_(r_lU, i_);

  const auto P = sibling_read_data_(sr_P);
  const auto pi = sibling_read_data_(sr_pi);
  const auto lV = sibling_read_data_(sr_lV);
  const auto lU = sibling_read_data_(sr_lU);
  const auto lW = sibling_read_data_(sr_lW);

  auto A = sibling_write_data_(sw_A);
  auto c = sibling_write_data_(sw_c);
  auto z = sibling_write_data_(sw_z);
  auto w = sibling_write_data_(sw_w);
  auto lQ = sibling_write_data_(sw_lQ);
  auto tau = sibling_write_data_(sw_tau);

  const int n = n_ - 1;
  auto lU_rsqrt = (lU_i + lU).rsqrt();
  c = (lV_i - lV) * lU_rsqrt;
  auto P_i_ratio = P_i / (P_i + P);
  z = kInvBeta * eigen_util::logit(P_i_ratio);  // TODO: try fast approximation
  z -= c;

  tau = compute_tau(lQ_i, lQ, lW_i, lW, z, lU_rsqrt);
  if (tau.isConstant(1.0f, 0.0f)) {
    // all tau are 1 - put all policy mass on this action
    full_write_data_(fw_A).fill(0.f);  // 0 means -inf
    full_write_data_(fw_A, i_) = -1.f;
    safety_check(__LINE__);
    return;
  } else if (tau.isZero(0.f)) {
    // all tau are 0 - put no policy mass on this action
    full_write_data_(fw_A, i_) = 0.f;  // 0 means -inf
    safety_check(__LINE__);
    return;
  }

  for (int j = 0; j < n; j++) {
    if (tau[j] == 1.0f) {
      // this action is dominated, zero its rating
      full_write_data_(fw_A, j >= i_ ? j + 1 : j) = 0.f;  // 0 means -inf
    }
  }

  w = P + kLambda * pi;
  solve_for_A_i(w, tau, A, A_i);
  full_write_data_(fw_A, i_) = A_i;
  safety_check(__LINE__);
}

// Add c to all nonzero A values to ensure max A is -1
template <search::concepts::Traits Traits>
void Backpropagator<Traits>::calibrate_ratings() {
  bool nonzero_found = false;
  float max_A = 0.f;

  for (int k = 0; k < n_; k++) {
    float A_k = full_write_data_(fw_A, k);
    if (A_k == 0.f) continue;
    if (nonzero_found) {
      if (A_k > max_A) {
        max_A = A_k;
      }
    } else {
      max_A = A_k;
      nonzero_found = true;
    }
  }

  RELEASE_ASSERT(nonzero_found, "All A values are zero - cannot calibrate ratings");
  float c = -1.f - max_A;
  for (int k = 0; k < n_; k++) {
    float& A_k = full_write_data_(fw_A, k);
    if (A_k != 0.f) {
      A_k += c;
    }
  }
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_policy() {
  // pi obtained by softmaxing the nonzero A values
  full_write_data_(fw_pi).fill(0.f);

  int m = 0;  // count nonzero
  for (int k = 0; k < n_; k++) {
    if (full_write_data_(fw_A, k) != 0.f) {
      m++;
    }
  }

  RELEASE_ASSERT(m > 0, "All A values are zero - cannot compute policy");
  LocalPolicyArray pi_nonzero(m);
  int w = 0;
  for (int k = 0; k < n_; k++) {
    if (full_write_data_(fw_A, k) != 0.f) {
      pi_nonzero(w++) = full_write_data_(fw_A, k);
    }
  }

  eigen_util::softmax_in_place(pi_nonzero);

  int r = 0;
  for (int k = 0; k < n_; k++) {
    if (full_write_data_(fw_A, k) != 0.f) {
      full_write_data_(fw_pi, k) = pi_nonzero(r++);
    }
  }
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

// Solves for A_i such that:
//   sum_j w[j] * (tau[j] - sigmoid(kBeta * (A_i - A[j]))) = 0
template <search::concepts::Traits Traits>
void Backpropagator<Traits>::solve_for_A_i(const LocalArray& w, const LocalArray& tau,
                                           const LocalArray& A, float& A_i) {
  RELEASE_ASSERT(!tau.isZero(0.f), "All tau values are zero - cannot solve for A_i");
  RELEASE_ASSERT(!tau.isConstant(1.f, 0.f), "All tau values are one - cannot solve for A_i");
  constexpr float kSat = math::detail::SigmoidLUT::kXMax;
  const int n = w.size();

  bool nonzero_found = false;
  float max_A = 0.f;
  float min_A = 0.f;
  for (int j = 0; j < n; ++j) {
    const float A_j = A[j];
    if (A_j == 0.f) continue;
    if (nonzero_found) {
      min_A = A_j < min_A ? A_j : min_A;
      max_A = A_j > max_A ? A_j : max_A;
    } else {
      min_A = A_j;
      max_A = A_j;
      nonzero_found = true;
    }
  }
  RELEASE_ASSERT(nonzero_found, "All A values are zero - cannot solve for A_i");
  float lo = std::min(A_i, min_A) - kSat * kInvBeta;
  float hi = std::max(A_i, max_A) + kSat * kInvBeta;

  // Fixed iterations: tune for accuracy/speed tradeoff. ChatGPT suggests 6-8.
  constexpr int kIters = 6;
  constexpr float tol = 1e-7f;  // TODO: this can probably be relaxed further

  Mask mask = (A != 0.f);  // skip -inf actions
  LocalArray A_m = eigen_util::mask_splice(A, mask);
  LocalArray w_m = eigen_util::mask_splice(w, mask);
  LocalArray tau_m = eigen_util::mask_splice(tau, mask);

  const float w_m_sum = w_m.sum();
  const float w_tau_m_sum = (w_m * tau_m).sum();

  for (int it = 0; it < kIters; ++it) {
    float F = 0.0f;   // sum w*(tau - s)
    float Sd = 0.0f;  // sum w*s*(1-s), used for derivative

    LocalArray s_m = eigen_util::sigmoid(kBeta * (A_i - A_m));  // TODO: try fast approximation
    LocalArray ws_m = w_m * s_m;
    F += w_tau_m_sum - ws_m.sum();
    Sd += (ws_m * (1.0f - s_m)).sum();

    if (std::abs(F) < tol * w_m_sum) break;

    // Maintain bracket via monotonicity:
    // F(c) decreases with c. If F>0 -> c too small -> move lo up. Else move hi down.
    if (F > 0.0f) {
      lo = A_i;
    } else {
      hi = A_i;
    }

    // Newton step with safeguard to bracket.
    // dF/dc = -kBeta * Sd  (negative). So Newton:
    //   c_new = c - F / dF = c + F / (kBeta*Sd)
    const float denom = kBeta * (Sd + 1e-12f);
    const float c_newton = A_i + F / denom;

    // Safeguard: if Newton jumps outside bracket, bisect.
    bool out_of_bounds = (c_newton < lo) || (c_newton > hi);
    A_i = out_of_bounds ? (0.5f * (lo + hi)) : c_newton;
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
  // auto lV = read_data_(r_lV);

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
void Backpropagator<Traits>::r_splice(read_col_t from_col, sibling_read_col_t to_col) {
  const auto from = read_data_(from_col);
  auto to = sibling_read_data_(to_col);

  if (i_ > 0) {
    to.topRows(i_) = from.topRows(i_);
  }

  int tail = n_ - i_ - 1;
  if (tail > 0) {
    to.bottomRows(tail) = from.bottomRows(tail);
  }
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::w_splice(full_write_col_t from_col, sibling_write_col_t to_col) {
  const auto from = full_write_data_(from_col);
  auto to = sibling_write_data_(to_col);

  if (i_ > 0) {
    to.topRows(i_) = from.topRows(i_);
  }

  int tail = n_ - i_ - 1;
  if (tail > 0) {
    to.bottomRows(tail) = from.bottomRows(tail);
  }
}

template <search::concepts::Traits Traits>
typename Backpropagator<Traits>::LocalArray Backpropagator<Traits>::unsplice(
  sibling_write_col_t from_col) {
  LocalArray result(n_);
  const auto from = sibling_write_data_(from_col);
  if (i_ > 0) {
    result.topRows(i_) = from.topRows(i_);
  }

  result[i_] = 0.f;  // unused

  int tail = n_ - i_ - 1;
  if (tail > 0) {
    result.bottomRows(tail) = from.bottomRows(tail);
  }
  return result;
}

template <search::concepts::Traits Traits>
template <typename T>
void Backpropagator<Traits>::normalize_policy(T pi) {
  float pi_i = pi(i_);
  float sum_others = pi.sum() - pi_i;

  if (sum_others > 0.f) {
    float mult = (1.f - pi_i) / sum_others;
    pi *= mult;
    pi(i_) = pi_i;
  } else {
    pi(i_) = 1.f;
  }
}

}  // namespace beta0
