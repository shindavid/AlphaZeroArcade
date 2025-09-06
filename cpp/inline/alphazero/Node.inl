#include "alphazero/Node.hpp"

namespace a0 {

template <core::concepts::Game Game>
void Node<Game>::Stats::init_q(const ValueArray& value, bool pure) {
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

template <core::concepts::Game Game>
void Node<Game>::Stats::update_provable_bits(const player_bitset_t& all_actions_provably_winning,
                                             const player_bitset_t& all_actions_provably_losing,
                                             int num_expanded_children, bool cp_has_winning_move,
                                             const StableData& sdata) {
  int num_valid_actions = sdata.num_valid_actions;
  core::seat_index_t seat = sdata.active_seat;

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

template <core::concepts::Game Game>
typename Node<Game>::Stats Node<Game>::stats_safe() const {
  // NOTE[dshin]: I attempted a version of this that attempted a lock-free read, resorting to a
  // the mutex only when a set dirty-bit was found on the copied stats. Contrary to my expectations,
  // this was slightly but clearly slower than the current version. I don't really understand why
  // this might be, but it's not worth investigating further at this time.
  mit::unique_lock lock(this->mutex());
  return stats_;
}

}  // namespace a0
