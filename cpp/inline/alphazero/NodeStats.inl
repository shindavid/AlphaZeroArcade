#include "alphazero/NodeStats.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::init_q(const ValueArray& value, bool pure) {
  Q = value;
  Q_sq = value * value;
  if (pure) {
    for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
      provably_winning[p] = value(p) >= Game::GameResults::kMaxValue;
      provably_losing[p] = value(p) <= Game::GameResults::kMinValue;
    }
  }

  eigen_util::debug_assert_is_valid_prob_distr(Q);
}

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::update_provable_bits(const player_bitset_t& all_actions_provably_winning,
                                               const player_bitset_t& all_actions_provably_losing,
                                               int num_expanded_children, bool cp_has_winning_move,
                                               int num_valid_actions, core::seat_index_t seat) {
  if (num_valid_actions == 0) {
    // terminal state, provably_winning/losing should already be set
  } else if (cp_has_winning_move) {
    provably_winning[seat] = true;
    provably_losing.set();
    provably_losing[seat] = false;
  } else if (num_expanded_children == num_valid_actions) {
    provably_winning = all_actions_provably_winning;
    provably_losing = all_actions_provably_losing;
  }
}

}  // namespace alpha0
