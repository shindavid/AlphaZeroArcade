#include "beta0/Backpropagator.hpp"

#include "beta0/Calculations.hpp"
#include "util/Gaussian1D.hpp"

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
  print_debug_info();
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
    }

    if (child) {
      const auto& lUV_k = child->stable_data().lUV[seat_];
      read_data_(r_lV, k) = lUV_k.mean();
      read_data_(r_lU, k) = lUV_k.variance();
    } else {
      const auto& lUV_k = child_edge->child_lAUV[seat_];
      read_data_(r_lV, k) = lUV_k.mean();
      read_data_(r_lU, k) = lUV_k.variance();
      read_data_(r_lQ, k) = read_data_(r_lV, k);
      read_data_(r_lW, k) = read_data_(r_lU, k);
      read_data2_(r2_Q, k) = child_edge->child_AV.transpose();
      read_data2_(r2_W, k) = child_edge->child_AU.transpose();
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

  splice(r_P, sr_P);
  splice(r_pi, sr_pi);
  splice(r_lV, sr_lV);
  splice(r_lU, sr_lU);
  splice(r_lQ, sr_lQ);
  splice(r_lW, sr_lW);
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::compute_update_rules() {
  full_write_data_.resize(n_);
  sibling_write_data_.resize(n_ - 1);
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
    child_edge->child_lQW = stats_.lQW[seat_];
  }

  int N = node_->stats().N;
  node_->stats() = stats_;  // copy back
  node_->stats().N = N;
  edge_->RC = N;
  if (edge_->XC > 0) {
    edge_->XC--;
  }
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::print_debug_info() {
  if (!search::kEnableSearchDebug) return;

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
  const LocalArray lV = read_data_(r_lV);
  const LocalArray lU = read_data_(r_lU);
  const LocalArray lQ = read_data_(r_lQ);
  const LocalArray lW = read_data_(r_lW);
  const LocalArray Q = read_data2_(r2_Q).col(seat_);
  const LocalArray W = read_data2_(r2_W).col(seat_);
  const LocalArray Q_star = read_data2_(r2_Q_star).col(seat_);

  const LocalArray pi_after = full_write_data_(fw_pi);

  const LocalArray c = unsplice(sw_c);
  const LocalArray z = unsplice(sw_z);
  const LocalArray tau_new = unsplice(sw_tau_new);
  const LocalArray tau_old = unsplice(sw_tau_old);

  LocalArray actions(n_);
  LocalArray i_indicator(n_);
  LocalArray N(n_);
  LocalArray lQ_old(n_);
  LocalArray lW_old(n_);

  auto sym = context_.root_canonical_sym;
  const auto& search_path = context_.search_path;
  for (const auto& visitation : search_path) {
    if (visitation.node == node_) {
      break;
    }
    sym = Game::SymmetryGroup::compose(sym, visitation.edge->sym);
  }

  group::element_t inv_sym = Game::SymmetryGroup::inverse(sym);
  for (int e = 0; e < n_; ++e) {
    auto edge = lookup_table().get_edge(node_, e);
    core::action_t action = edge->action;
    Game::Symmetries::apply(action, inv_sym, node_->action_mode());
    actions(e) = action;
    i_indicator(e) = (e == i_) ? 1.f : 0.f;
    lQ_old(e) = (e == i_) ? lQW_i_old_.mean() : 0;
    lW_old(e) = (e == i_) ? lQW_i_old_.variance() : 0;

    auto child = lookup_table().get_node(edge->child_index);
    N(e) = child ? child->stats().N : 0;
  }

  static std::vector<std::string> action_columns = {
    "action", "i", "N", "P",  "pi", "lV", "lU",      "lQo",     "lWo", "lQ",
    "lW",     "Q", "W", "Q*", "c",  "z",  "tau_old", "tau_new", "PI"};


  auto action_data = eigen_util::sort_rows(
    eigen_util::concatenate_columns(actions, i_indicator, N, P, pi_before, lV, lU, lQ_old, lW_old,
                                    lQ, lW, Q, W, Q_star, c, z, tau_old, tau_new, pi_after));

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
void Backpropagator<Traits>::compute_policy() {
  const util::Gaussian1D lQW_i(read_data_(r_lQ, i_), read_data_(r_lW, i_));
  if (lQW_i == util::Gaussian1D::neg_inf()) {
    full_write_data_(fw_pi) = read_data_(r_pi);
    full_write_data_(fw_pi, i_) = 0.f;
    normalize_policy(full_write_data_(fw_pi));
    return;
  } else if (lQW_i == util::Gaussian1D::pos_inf()) {
    full_write_data_(fw_pi).fill(0.f);
    full_write_data_(fw_pi, i_) = 1.f;
    return;
  }

  if (read_data_(r_lW).isZero()) {
    // all actions have zero variance - put all policy mass on the best action(s)
    float max_lQ = read_data_(r_lQ).maxCoeff();
    full_write_data_(fw_pi).fill(0.f);
    for (int k = 0; k < n_; k++) {
      if (read_data_(r_lQ, k) == max_lQ) {
        full_write_data_(fw_pi, k) = 1.f;
      }
    }
    normalize_policy(full_write_data_(fw_pi));
    return;
  }

  RELEASE_ASSERT(lQW_i.valid());

  full_write_data_(fw_pi) = read_data_(r_pi);

  if (read_data_(r_pi, i_) >= 1.f) {
    // all policy mass is already on this action
    return;
  }

  const float P_i = read_data_(r_P, i_);
  const float pi_i = read_data_(r_pi, i_);
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

  const int n = n_ - 1;

  auto c = sibling_write_data_(sw_c);
  auto z = sibling_write_data_(sw_z);
  auto tau_new = sibling_write_data_(sw_tau_new);
  auto tau_old = sibling_write_data_(sw_tau_old);

  c = (lV_i - lV) / (lU_i + lU).sqrt();
  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(P_i, P.data(), c.data(), n, z.data());
  z -= c;

  tau_new = compute_tau(lQ_i, lQ, lW_i, lW, z);
  tau_old = compute_tau(lQW_i_old_.mean(), lQ, lQW_i_old_.variance(), lW, z);

  const float pi_i_inv = 1.0f / (1.0f - pi_i);
  float pi_i_new = (pi * tau_new).sum() * pi_i_inv;
  float pi_i_old = (pi * tau_old).sum() * pi_i_inv;

  float pi_i_new_logit = math::fast_coarse_logit(pi_i_new);
  float pi_i_old_logit = math::fast_coarse_logit(pi_i_old);
  float delta_logit = pi_i_new_logit - pi_i_old_logit;

  float pi_i_logit = math::fast_coarse_logit(pi_i);
  pi_i_logit += delta_logit;
  full_write_data_(fw_pi, i_) = math::fast_coarse_sigmoid(pi_i_logit);

  normalize_policy(full_write_data_(fw_pi));
}

template <search::concepts::Traits Traits>
typename Backpropagator<Traits>::LocalArray Backpropagator<Traits>::compute_tau(
  float lQ_i, const LocalArray& lQ, float lW_i, const LocalArray& lW, const LocalArray& z) {
  // Effectively computes:
  //
  // return (lQ_i - lQ) / (lW_i + lW).sqrt();
  //
  // But does some special casing for infinities and zero variances. Each (lQ, lW) pair comes from
  // a util::Gaussian1D, so we need to handle the various edge cases corresponding to how
  // util::Gaussian1D represents +/- infinity.

  if (lW_i == util::Gaussian1D::kVarianceNegInf) {
    return lQ * 0;
  } else if (lW_i == util::Gaussian1D::kVariancePosInf) {
    return lQ * 0 + 1.0f;
  } else {
    LocalArray S = (lQ_i - lQ) / (lW_i + lW).sqrt() + z;

    int n = S.size();
    for (int k = 0; k < n; k++) {
      float lW_k = lW(k);
      if (lW_k == util::Gaussian1D::kVarianceNegInf) {
        S[k] = -999.f;
      } else if (lW_k == util::Gaussian1D::kVariancePosInf) {
        S[k] = +999.f;
      } else if (lW_i == 0.f && lW_k == 0.f) {
        if (lQ_i > lQ(k)) {
          S[k] = +999.f;
        } else if (lQ_i < lQ(k)) {
          S[k] = -999.f;
        } else {
          S[k] = z[k];
        }
      }
    }

    LocalArray tau = S;
    math::fast_coarse_batch_normal_cdf(S.data(), n, tau.data());
    return tau;
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

  auto pi = full_write_data_(fw_pi);
  auto W = read_data2_(r2_W);
  auto Q_star = read_data2_(r2_Q_star);
  auto pi_mat = pi.matrix();
  auto Q = read_data2_(r2_Q);

  stats_.Q = (Q_star.matrix().transpose() * pi_mat).array();

  // check that stats_.Q is >= 0 everywhere, dump it and abort if not
  for (int p = 0; p < kNumPlayers; ++p) {
    if (stats_.Q(p) < Game::GameResults::kMinValue) {
      std::ostringstream ss;
      ss << "Backpropagated Q value is invalid: " << stats_.Q.transpose()
         << "\nQ_foor_: " << Q_floor_ << "\nseat_: " << int(seat_) << "\ni_:" << i_ << "\nQ:\n"
         << Q << "\nQ_star:\n"
         << Q_star << "\nW:\n"
         << W << "\npi:\n"
         << pi.transpose();
      LOG_ERROR(ss.str());
      throw std::runtime_error("Invalid Q value in Backpropagator");
    }
  }

  auto W_in_mat = W.matrix();
  auto W_across_mat = (Q_star.rowwise() - stats_.Q.transpose()).square().matrix();
  auto W_p_mat = (W_in_mat + W_across_mat).transpose() * pi_mat;

  stats_.W = W_p_mat.array();

  Calculations<Traits>::populate_logit_value_beliefs(stats_.Q, stats_.W, stats_.lQW);
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::splice(read_col_t from_col, sibling_read_col_t to_col) {
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
