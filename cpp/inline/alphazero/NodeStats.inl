#include "alphazero/NodeStats.hpp"

namespace alpha0 {

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::update_q(const ValueArray& q, const ValueArray& q_sq, bool pure) {
  Q = q;
  Q_sq = q_sq;
  if (pure) {
    for (int p = 0; p < Game::Constants::kNumPlayers; ++p) {
      provably_winning[p] = Q(p) >= Game::GameResults::kMaxValue;
      provably_losing[p] = Q(p) <= Game::GameResults::kMinValue;
    }
  }
}

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::update_provable_bits(const player_bitset_t& all_actions_provably_winning,
                                               const player_bitset_t& all_actions_provably_losing,
                                               bool cp_has_winning_move, bool all_edges_expanded,
                                               core::seat_index_t seat) {
  if (cp_has_winning_move) {
    provably_winning[seat] = true;
    provably_losing.set();
    provably_losing[seat] = false;
  } else if (all_edges_expanded) {
    provably_winning = all_actions_provably_winning;
    provably_losing = all_actions_provably_losing;
  }
}

}  // namespace alpha0
