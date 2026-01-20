#include "beta0/Backpropagator.hpp"

#include "beta0/Calculations.hpp"
#include "beta0/Constants.hpp"
#include "search/Constants.hpp"
#include "util/EigenUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/Gaussian1D.hpp"
#include "util/Math.hpp"

#include <sstream>

namespace beta0 {

static constexpr float kBeta = 1.702f;  // logistic approximation constant
static constexpr float kInvBeta = 1.0f / kBeta;

static constexpr float kLambda = 0.f;  // pi contribution to w_{ij}
static constexpr bool kDisableZMargining = false;

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
  read_data2_(r2_Q, k) = child_stats.Q.transpose();
  read_data2_(r2_W, k) = child_stats.W.transpose();
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
  read_data2_.resize(n_);

  int i = -1;
  for (int k = 0; k < n_; k++) {
    const Edge* child_edge = lookup_table().get_edge(node_, k);
    const Node* child = lookup_table().get_node(child_edge->child_index);

    if (child_edge == edge_) {
      i = k;
      child_i_ = child;
    }

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
      read_data2_(r2_Q, k) = child_edge->child_AV.transpose();
      read_data2_(r2_W, k) = child_edge->child_AU.transpose();
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
  lQW_i_old_ = edge_->child_lQW;

  for (int k = 0; k < n_; k++) {
    const Edge* child_edge = lookup_table().get_edge(node_, k);
    read_data_(r_P, k) = child_edge->P;
    read_data_(r_pi, k) = child_edge->pi;
    read_data_(r_A, k) = child_edge->A;

    const Node* child = lookup_table().get_node(child_edge->child_index);
    if (child) {
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
  // 1. Read children data that we deferred earlier to avoid deadlocks.

  for (int c = 0; c < num_deferred_child_stats_load_indices_; c++) {
    int k = deferred_child_stats_load_indices_[c];
    const Edge* child_edge = lookup_table().get_edge(node_, k);
    const Node* child = lookup_table().get_node(child_edge->child_index);

    const auto child_stats = child->stats_safe();  // make a copy - acquires child mutex
    load_child_stats(k, child_stats);
  }

  // 2. Compute Q_floor_

  Q_floor_ = Game::GameResults::kMinValue;
  const auto Q = read_data2_(r2_Q);
  const auto W = read_data2_(r2_W);
  for (int k = 0; k < n_; ++k) {
    if (W(k, seat_) == 0.f) {
      Q_floor_ = std::max(Q_floor_, Q(k, seat_));
    }
  }

  // 3. Compute Q_star

  auto Q_star = read_data2_(r2_Q_star);

  Q_star = Q;
  if (Q_floor_ > Game::GameResults::kMinValue) {
    // Cap by Q_floor where necessary
    for (int k = 0; k < n_; ++k) {
      if (W(k, seat_) == 0.f) {
        continue;
      }
      float Q_k = Q(k, seat_);
      if (Q_k < Q_floor_) {
        modify_Q_arr(Q_star, k, seat_, Q_floor_);
      }
    }
  }

  // 4. Copy read data into sibling_read_data_

  sibling_read_data_.resize(n_ - 1);

  r_splice(r_P, sr_P);
  r_splice(r_pi, sr_pi);
  r_splice(r_lV, sr_lV);
  r_splice(r_lU, sr_lU);
  r_splice(r_lQ, sr_lQ);
  r_splice(r_lW, sr_lW);
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_update_rules() {
  full_write_data_.resize(n_);
  sibling_write_data_.resize(n_ - 1);
  compute_ratings();
  calibrate_ratings();
  compute_policy();
  update_QW();

  stats_.Q_min = stats_.Q_min.cwiseMin(stats_.Q);
  stats_.Q_max = stats_.Q_max.cwiseMax(stats_.Q);
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::apply_updates() {
  mit::unique_lock lock(node_->mutex());

  for (int k = 0; k < n_; k++) {
    Edge* child_edge = lookup_table().get_edge(node_, k);
    child_edge->pi = full_write_data_(fw_pi, k);
    child_edge->A = full_write_data_(fw_A, k);
    child_edge->child_lQW = stats_.lQW[seat_];
  }

  int N = node_->stats().N;
  node_->stats() = stats_;  // copy back
  node_->stats().N = N;
  edge_->RC = N;
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::print_debug_info() {
  std::ostringstream ss;
  ss << std::format("{:>{}}", "", context_.log_prefix_n());

  ValueArray players;
  ValueArray nQ = stats_.Q;
  ValueArray CP;
  for (int p = 0; p < kNumPlayers; ++p) {
    players(p) = p;
    CP(p) = p == seat_;
  }

  static std::vector<std::string> player_columns = {"Seat", "Q", "CurP"};
  auto player_data = eigen_util::concatenate_columns(players, nQ, CP);

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

  const LocalArray P = read_data_(r_P);
  const LocalArray pi_before = read_data_(r_pi);
  const LocalArray A_before = read_data_(r_A);
  const LocalArray lV = read_data_(r_lV);
  const LocalArray lU = read_data_(r_lU);
  const LocalArray lQ = read_data_(r_lQ);
  const LocalArray lW = read_data_(r_lW);
  const LocalArray Q = read_data2_(r2_Q).col(seat_);
  const LocalArray W = read_data2_(r2_W).col(seat_);
  const LocalArray Q_star = read_data2_(r2_Q_star).col(seat_);

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

  static std::vector<std::string> action_columns = {"action", "i",  "N",   "P",   "pi", "A",  "lV",
                                                    "lU",     "lQ", "lW",  "Q",   "W",  "Q*", "c",
                                                    "z",      "w",  "tau", "pi*", "A*"};

  auto action_data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions, i_indicator, N, P, pi_before, A_before, lV, lU, lQ, lW,
                                    Q, W, Q_star, c, z, w, tau, pi_after, A_after));

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
void Backpropagator<Traits>::compute_ratings() {
  const util::Gaussian1D lQW_i(read_data_(r_lQ, i_), read_data_(r_lW, i_));
  full_write_data_(fw_A) = read_data_(r_A);
  if (handle_edge_cases()) return;

  if (lQW_i == util::Gaussian1D::neg_inf()) return;
  if (read_data_(r_pi, i_) >= 1.f) return;  // all policy mass is already on this action

  w_splice(fw_A, sw_A);

  RELEASE_ASSERT(lQW_i.valid());

  const float P_i = read_data_(r_P, i_);
  float A_i = read_data_(r_A, i_);
  const float lQ_i = read_data_(r_lQ, i_);
  const float lW_i = read_data_(r_lW, i_);
  const float lV_i = read_data_(r_lV, i_);
  const float lU_i = read_data_(r_lU, i_);

  const auto P = sibling_read_data_(sr_P);
  const auto pi = sibling_read_data_(sr_pi);
  const auto lV = sibling_read_data_(sr_lV);
  const auto lU = sibling_read_data_(sr_lU);
  const auto lQ = sibling_read_data_(sr_lQ);
  const auto lW = sibling_read_data_(sr_lW);

  auto A = sibling_write_data_(sw_A);
  auto c = sibling_write_data_(sw_c);
  auto z = sibling_write_data_(sw_z);
  auto w = sibling_write_data_(sw_w);
  auto tau = sibling_write_data_(sw_tau);

  const int n = n_ - 1;
  if (kDisableZMargining) {
    z.setZero();
  } else {
    c = (lV_i - lV) / (lU_i + lU).sqrt();
    auto P_i_ratio = P_i / (P_i + P);
    for (int j = 0; j < n; j++) {
      z[j] = kInvBeta * math::fast_coarse_logit(P_i_ratio[j]);
    }
    z -= c;
    // TODO: replace above with a clamped calculation for z
  }

  tau = compute_tau(lQ_i, lQ, lW_i, lW, z);
  if (tau.isConstant(1.0f, 0.0f)) {
    // all tau are 1 - put all policy mass on this action
    full_write_data_(fw_A).fill(0.f);  // 0 means -inf
    full_write_data_(fw_A, i_) = -1.f;
    return;
  } else if (tau.isZero(0.f)) {
    // all tau are 0 - put no policy mass on this action
    full_write_data_(fw_A, i_) = 0.f;  // 0 means -inf
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
  float lQ_i, const LocalArray& lQ, float lW_i, const LocalArray& lW, const LocalArray& z) {
  // Effectively computes:
  //
  // sigmoid(kBeta [ (lQ_i - lQ) / (lW_i + lW).sqrt() + z ])
  //
  // But does some special casing for infinities and zero variances. Each (lQ, lW) pair comes from
  // a util::Gaussian1D, so we need to handle the various edge cases corresponding to how
  // util::Gaussian1D represents +/- infinity.

  if (lW_i == util::Gaussian1D::kVarianceNegInf) {
    return lQ * 0;
  } else if (lW_i == util::Gaussian1D::kVariancePosInf) {
    return lQ * 0 + 1.0f;
  } else {
    int n = z.size();
    LocalArray tau(n);
    for (int j = 0; j < n; j++) {
      float lW_j = lW(j);
      if (lW_j == util::Gaussian1D::kVarianceNegInf) {
        tau[j] = 1.0f;
      } else if (lW_j == util::Gaussian1D::kVariancePosInf) {
        tau[j] = 0.0f;
      } else if (lW_i == 0.f && lW_j == 0.f) {
        if (lQ_i > lQ(j)) {
          tau[j] = 1.0f;
        } else if (lQ_i < lQ(j)) {
          tau[j] = 0.0f;
        } else {
          tau[j] = math::fast_coarse_sigmoid(kBeta * z[j]);
        }
      } else {
        float lQ_j = lQ(j);
        float S_j = kBeta * ((lQ_i - lQ_j) / std::sqrt(lW_i + lW_j) - z[j]);
        tau[j] = math::fast_coarse_sigmoid(S_j);
      }
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

  for (int it = 0; it < kIters; ++it) {
    float F = 0.0f;   // sum w*(tau - s)
    float Sd = 0.0f;  // sum w*s*(1-s), used for derivative
    float sw = 0.0f;  // sum w

    for (int j = 0; j < n; ++j) {
      float A_j = A[j];
      if (A_j == 0.f) continue;  // skip -inf actions

      const float x = kBeta * (A_i - A_j);
      const float s = math::fast_coarse_sigmoid(x);
      const float wj = w[j];

      F += wj * (tau[j] - s);
      Sd += wj * (s * (1.0f - s));
      sw += wj;
    }

    if (std::abs(F) < 1e-7f * sw) break;

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
void Backpropagator<Traits>::update_QW() {
  // Q/W-Update rules:
  //
  // Q(p) = sum_i pi_i * Q^*_i
  // W(p) = sum_i pi_i (W_i + (Q^*_i - Q(p))^2)
  //
  // where Q^*_i is the *conditional* belief
  //
  // Q^*_i = E[Z_i | i = argmax_j Z_j]
  //
  // We approximate Q^*_i = max(Q_i, Q_floor), where
  //
  // Q_floor is the maximum Q_k over all actions k with W_k = 0 (i.e. no uncertainty).

  ComputationCheckMethod method = kAllowInf;

  auto pi = full_write_data_(fw_pi);
  auto W = read_data2_(r2_W);
  auto Q_star = read_data2_(r2_Q_star);

  float Q_k_check = -1.f;
  float pi_k_check = -1.f;
  int k_check = -1;
  for (int k = 0; k < n_; ++k) {
    float Q_k = Q_star(k, seat_);
    float pi_k = pi(k);
    if (pi_k > 0.f) {
      if (Q_k > Game::GameResults::kMinValue && Q_k < Game::GameResults::kMaxValue) {
        method = kAssertFinite;
        k_check = k;
        pi_k_check = pi_k;
        Q_k_check = Q_k;
        break;
      }
    }
  }

  Calculations<Game>::dot_product(Q_star, pi, stats_.Q);

  auto W_in_mat = W.matrix();
  auto W_across_mat = (Q_star.rowwise() - stats_.Q.transpose()).square().matrix();
  Calculations<Game>::dot_product(W_in_mat + W_across_mat, pi, stats_.W);

  try {
    Calculations<Game>::populate_logit_value_beliefs(stats_.Q, stats_.W, stats_.lQW, method);
  } catch (util::ReleaseAssertionError& e) {
    LOG_ERROR("Backpropagator Q/W update failed computation check");
    print_debug_info();
    LOG_ERROR(" Q_k_check:  {}", Q_k_check);
    LOG_ERROR(" pi_k_check: {}", pi_k_check);
    LOG_ERROR(" k_check:    {}", k_check);
    LOG_ERROR(" pi:         {}", fmt::streamed(pi.transpose()));
    LOG_ERROR(" Q*:         {}", fmt::streamed(Q_star.transpose()));
    LOG_ERROR(" W:          {}", fmt::streamed(W.transpose()));
    LOG_ERROR(" stats_.Q:   {}", fmt::streamed(stats_.Q.transpose()));
    LOG_ERROR(" stats_.W:   {}", fmt::streamed(stats_.W.transpose()));
    throw e;
  }
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
void Backpropagator<Traits>::modify_Q_arr(T& Q_arr, int action_index, core::seat_index_t seat,
                                          float q_new) {
  int k = action_index;
  float Q_k = Q_arr(k, seat);
  auto row_k = Q_arr.row(k);
  float delta = q_new - Q_k;

  if constexpr (kNumPlayers == 1) {
    // In single-player, we can just adjust the Q value directly.
    row_k(seat) = q_new;
  } else if constexpr (kNumPlayers == 2) {
    // In two-player zero-sum, we can just adjust both players' Q values symmetrically.
    row_k(seat) = q_new;
    row_k(1 - seat) -= delta;
    if (row_k(1 - seat) < Game::GameResults::kMinValue) {
      row_k(1 - seat) = Game::GameResults::kMinValue;
    }
  } else {
    // For multiplayer games, it's a little more ambiguous how we should adjust the other
    // players' Q values.
    //
    // Here, we scale the other players' Q values by a constant chosen such that the sum of Q
    // values remains the same after the adjustment.
    float Q_sum = row_k.sum();
    float mult = (Q_sum - delta) / Q_sum;
    row_k *= mult;
    row_k(seat) = q_new;
  }
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
