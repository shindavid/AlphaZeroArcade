#include "games/stochastic_nim/players/PerfectPlayer.hpp"

#include "games/stochastic_nim/Constants.hpp"
#include "games/stochastic_nim/Move.hpp"

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
  if (request.aux) {
    return Move(request.aux - 1, stochastic_nim::kPlayerPhase);
  }

  const State& state = request.info_set;
  const MoveSet& valid_moves = request.valid_moves;
  RELEASE_ASSERT(state.phase == kPlayerPhase);

  ActionResponse response;

  Move move;
  if (params_.strength == 0) {
    move = valid_moves.get_random(util::Random::default_prng());
  } else {
    move = Move(strategy_->get_optimal_action(state.stones_left), stochastic_nim::kPlayerPhase);
  }
  response.set_move(move);
  response.set_aux(move.index() + 1);
  return response;
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
