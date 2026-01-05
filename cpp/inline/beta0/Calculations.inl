#include "beta0/Calculations.hpp"

#include "util/Math.hpp"

namespace beta0 {

template <search::concepts::Traits Traits>
void Calculations<Traits>::populate_logit_value_beliefs(const ValueArray& Q,
                                                       const ValueArray& W,
                                                       LogitValueArray& lQW) {
  if (kNumPlayers == 2) {
    // In this case, we only need to compute for one player, since the other is just negation.
    lQW[0] = compute_logit_value_belief(Q[0], W[0]);
    lQW[1] = -lQW[0];
  } else {
    for (core::seat_index_t p = 0; p < kNumPlayers; ++p) {
      lQW[p] = compute_logit_value_belief(Q[p], W[p]);
    }
  }
}

template <search::concepts::Traits Traits>
util::Gaussian1D Calculations<Traits>::compute_logit_value_belief(float Q, float W) {
  constexpr float kMin = Game::GameResults::kMinValue;
  constexpr float kMax = Game::GameResults::kMaxValue;
  constexpr float kWidth = kMax - kMin;
  constexpr float kInvWidth = 1.0f / kWidth;

  if (Q <= kMin) {
    return util::Gaussian1D::neg_inf();
  } else if (Q >= kMax) {
    return util::Gaussian1D::pos_inf();
  }
  if (W == 0) {
    float theta = math::fast_coarse_logit((Q - kMin) * kInvWidth);
    return util::Gaussian1D(theta, 0.f);
  }

  float mu = Q;
  float sigma_sq = W;

  // Rescale Q and W to reflect [0, 1] range
  mu = (mu - kMin) * kInvWidth;
  sigma_sq *= kInvWidth * kInvWidth;

  float mult = 1.0f / (mu * mu * (1 - mu) * (1 - mu));

  float theta1 = math::fast_coarse_logit(mu);
  float theta2 = (0.5 - mu) * sigma_sq * mult;
  float theta = theta1 - theta2;

  float omega_sq = sigma_sq * mult;
  return util::Gaussian1D(theta, omega_sq);
}


}  // namespace beta0
