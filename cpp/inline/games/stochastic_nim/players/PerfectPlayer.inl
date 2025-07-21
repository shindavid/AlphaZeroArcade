#include "games/stochastic_nim/players/PerfectPlayer.hpp"

namespace stochastic_nim {

inline auto PerfectPlayer::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("stochastic_nim::PerfectPlayer options");
  return desc
    .template add_option<"strength", 's'>(po::value<int>(&strength)->default_value(strength),
                                          "strength (0-1). 0 is random, 1 is perfect.")
    .template add_option<"verbose", 'v'>(po::bool_switch(&verbose)->default_value(verbose),
                                         "verbose mode");
}

inline PerfectPlayer::ActionResponse PerfectPlayer::get_action_response(
  const ActionRequest& request) {
  const State& state = request.state;
  const ActionMask& valid_actions = request.valid_actions;
  RELEASE_ASSERT(state.current_mode == kPlayerMode);

  if (params_.strength == 0) {
    return bitset_util::choose_random_on_index(valid_actions);
  }
  return strategy_->get_optimal_action(state.stones_left);
}

PerfectStrategy::PerfectStrategy() {
  init_boundary_conditions();
  iterate();
}

void PerfectStrategy::init_boundary_conditions() {
  Qa[0] = 0.0;
  Qb[0] = 1.0;
}

inline int PerfectStrategy::get_optimal_action(int stones_left) const {
  RELEASE_ASSERT((stones_left <= stochastic_nim::kStartingStones) && (stones_left > 0));
  return P[stones_left] - 1;
}

inline void PerfectStrategy::iterate() {
  constexpr int n = stochastic_nim::kStartingStones;
  constexpr int c = stochastic_nim::kChanceDistributionSize;
  Eigen::Vector<float, c> probs{stochastic_nim::kChanceEventProbs};
  auto arange_c = Eigen::ArrayXi::LinSpaced(c, 0, c - 1);
  for (int k = 1; k <= n; k++) {
    int m = std::min(k, stochastic_nim::kMaxStonesToTake);

    P[k] = m - eigen_util::argmax(Qb.segment(k - m, m));
    Qa[k] = Qb[k - P[k]];
    Qb[k] = 1.0 - probs.dot(eigen_util::slice(Qa, (k - arange_c).cwiseMax(0)));
  }
}

}  // namespace stochastic_nim
