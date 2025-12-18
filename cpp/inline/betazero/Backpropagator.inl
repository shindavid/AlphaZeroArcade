#include "betazero/Backpropagator.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
Backpropagator<Traits>::Backpropagator(SearchContext& context, Node* node, Edge* edge)
    : context_(context), node_(node), edge_(edge) {
  RELEASE_ASSERT(!node_->stable_data().is_chance_node, "Chance nodes not yet supported");

  preload_parent_data();
  load_parent_data();
  load_remaining_data();
  compute_update_rules();
  apply_updates();
}

template <search::concepts::Traits Traits>
bool Backpropagator<Traits>::shares_mutex_with_parent(
  const Node* child) const {
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
void Backpropagator<Traits>::load_parent_data() {
  // Now read data that requires locking the parent mutex.

  mit::unique_lock lock(node_->mutex());

  stats_ = node_->stats();  // copy

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
  stats_.W_max = stats_.W_max.cwiseMax(stats_.W);
}

template <search::concepts::Traits Traits>
void Backpropagator<Traits>::apply_updates() {
  mit::unique_lock lock(node_->mutex());

  for (int k = 0; k < n_; k++) {
    Edge* child_edge = lookup_table().get_edge(node_, k);
    child_edge->pi = full_write_data_(fw_pi, k);
  }

  node_->stats() = stats_;  // copy back
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

  auto S_denom_inv = sibling_write_data_(sw_S_denom_inv);
  auto S = sibling_write_data_(sw_S);
  auto c = sibling_write_data_(sw_c);
  auto z = sibling_write_data_(sw_z);
  auto tau = sibling_write_data_(sw_tau);

  c = (lV_i - lV) / (lU_i + lU).sqrt();
  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(P_i, P.data(), c.data(), n, z.data());
  z -= c;

  S_denom_inv = 1.0f / (lW_i + lW).sqrt();
  const float pi_i_inv = 1.0f / pi_i;

  S = (lQ_i - lQ) * S_denom_inv + z;
  math::fast_coarse_batch_normal_cdf(S.data(), n, tau.data());
  full_write_data_(fw_pi, i_) = (pi * tau).sum() * pi_i_inv;
  normalize_policy(full_write_data_(fw_pi));
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

  stats_.Q = (Q_star.matrix().transpose() * pi_mat).array();

  auto W_in_mat = W.matrix();
  auto W_across_mat = (Q_star.rowwise() - stats_.Q.transpose()).square().matrix();
  auto W_p_mat = (W_in_mat + W_across_mat).transpose() * pi_mat;

  stats_.W = W_p_mat.array();
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
template <typename T>
void Backpropagator<Traits>::modify_Q_arr(T& Q_arr, int action_index,
                                          core::seat_index_t seat, float q_new) {
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
