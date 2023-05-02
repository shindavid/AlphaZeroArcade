#include <othello/GameState.hpp>

#include <algorithm>
#include <bit>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <util/AnsiCodes.hpp>
#include <util/BitSet.hpp>
#include <util/CppUtil.hpp>

inline std::size_t std::hash<othello::GameState>::operator()(const othello::GameState& state) const {
  return state.hash();
}

namespace othello {

inline size_t GameState::serialize_action(char* buffer, size_t buffer_size, common::action_index_t action) {
  size_t n = snprintf(buffer, buffer_size, "%d", action);
  if (n >= buffer_size) {
    throw util::Exception("Buffer too small (%ld >= %ld)", n, buffer_size);
  }
  return n;
}

inline void GameState::deserialize_action(const char* buffer, common::action_index_t* action) {
  *action = boost::lexical_cast<int>(buffer);

  if (*action < 0 || *action >= othello::kNumGlobalActions) {
    throw util::Exception("Invalid action \"%s\" (action=%d)", buffer, *action);
  }
}

inline size_t GameState::serialize_state_change(
    char* buffer, size_t buffer_size, common::seat_index_t seat, common::action_index_t action) const
{
  return serialize_action(buffer, buffer_size, action);
}

inline void GameState::deserialize_state_change(
    const char* buffer, common::seat_index_t* seat, common::action_index_t* action)
{
  *seat = get_current_player();
  deserialize_action(buffer, action);
  apply_move(*action);
}

inline size_t GameState::serialize_game_end(char* buffer, size_t buffer_size, const GameOutcome& outcome) const {
  size_t n = 0;
  bool b = outcome[kBlack] > 0;
  bool w = outcome[kWhite] > 0;
  n += snprintf(buffer + n, buffer_size - n, b ? "B" : "");
  n += snprintf(buffer + n, buffer_size - n, w ? "W" : "");
  if (n >= buffer_size) {
    throw util::Exception("Buffer too small (%ld >= %ld)", n, buffer_size);
  }
  return n;
}

inline void GameState::deserialize_game_end(const char* buffer, GameOutcome* outcome) {
  outcome->setZero();
  const char* c = buffer;
  while (*c != '\0') {
    switch (*c) {
      case 'B': (*outcome)(kBlack) = 1; break;
      case 'W': (*outcome)(kWhite) = 1; break;
      default: throw util::Exception(R"(Invalid game end "%c" parsed from "%s")", *c, buffer);
    }
    ++c;
  }

  *outcome /= outcome->sum();
}

// copied from edax-reversi repo - board_next()
inline common::GameStateTypes<GameState>::GameOutcome GameState::apply_move(common::action_index_t action) {
  if (action == kPass) {
    std::swap(cur_player_mask_, opponent_mask_);
    cur_player_ = 1 - cur_player_;
    pass_count_++;
    if (pass_count_ == kNumPlayers) {
      return compute_outcome();
    }
  } else {
    mask_t flipped = flip[action](cur_player_mask_, opponent_mask_);
    mask_t cur_player_mask = opponent_mask_ ^ flipped;

    opponent_mask_ = cur_player_mask_ ^ (flipped | (1ULL << action));
    cur_player_mask_ = cur_player_mask;
    cur_player_ = 1 - cur_player_;
    pass_count_ = 0;

    if ((opponent_mask_ | cur_player_mask_) == kCompleteBoardMask) {
      return compute_outcome();
    }
  }

  GameOutcome outcome;
  outcome.setZero();
  return outcome;
}

inline GameState::ActionMask GameState::get_valid_actions() const {
  mask_t mask = get_moves(cur_player_mask_, opponent_mask_);
  if (mask == 0) {
    mask = 1ULL << kPass;
  }
  return reinterpret_cast<ActionMask&>(mask);
}

template<eigen_util::FixedTensorConcept InputSlab> void GameState::tensorize(InputSlab& tensor) const {
  throw util::Exception("TODO");
}

inline void GameState::dump(common::action_index_t last_action, const player_name_array_t* player_names) const {
  throw util::Exception("TODO");
}

inline std::size_t GameState::hash() const {
  return util::tuple_hash(to_tuple());
}

inline void GameState::dump_mcts_output(
    const ValueProbDistr& mcts_value, const LocalPolicyProbDistr& mcts_policy, const MctsResults& results)
{
  throw util::Exception("TODO");
}

inline void GameState::row_dump(row_t row, column_t blink_column) const {
  throw util::Exception("TODO");
}

inline typename GameState::GameOutcome GameState::compute_outcome() const {
  GameOutcome outcome;
  outcome.setZero();

  int opponent_count = std::popcount(opponent_mask_);
  int cur_player_count = std::popcount(cur_player_mask_);
  if (cur_player_count > opponent_count) {
    outcome(cur_player_) = 1;
  } else if (opponent_count > cur_player_count) {
    outcome(1 - cur_player_) = 1;
  } else {
    outcome.setConstant(0.5);
  }

  return outcome;
}

// copied from edax-reversi repo
inline mask_t GameState::get_moves(mask_t P, mask_t O) {
  mask_t mask = O & 0x7E7E7E7E7E7E7E7Eull;

  return (get_some_moves(P, mask, 1) // horizontal
          | get_some_moves(P, O, 8)   // vertical
          | get_some_moves(P, mask, 7)   // diagonals
          | get_some_moves(P, mask, 9))
         & ~(P|O); // mask with empties
}

// copied from edax-reversi repo
inline mask_t GameState::get_some_moves(mask_t P, mask_t mask, int dir) {
#if PARALLEL_PREFIX & 1
  // 1-stage Parallel Prefix (intermediate between kogge stone & sequential)
    // 6 << + 6 >> + 7 | + 10 &
    register unsigned long long flip_l, flip_r;
    register unsigned long long mask_l, mask_r;
    const int dir2 = dir + dir;

    flip_l  = mask & (P << dir);          flip_r  = mask & (P >> dir);
    flip_l |= mask & (flip_l << dir);     flip_r |= mask & (flip_r >> dir);
    mask_l  = mask & (mask << dir);       mask_r  = mask & (mask >> dir);
    flip_l |= mask_l & (flip_l << dir2);  flip_r |= mask_r & (flip_r >> dir2);
    flip_l |= mask_l & (flip_l << dir2);  flip_r |= mask_r & (flip_r >> dir2);

    return (flip_l << dir) | (flip_r >> dir);

#elif KOGGE_STONE & 1
  // kogge-stone algorithm
    // 6 << + 6 >> + 12 & + 7 |
    // + better instruction independency
    register unsigned long long flip_l, flip_r;
    register unsigned long long mask_l, mask_r;
    const int dir2 = dir << 1;
    const int dir4 = dir << 2;

    flip_l  = P | (mask & (P << dir));    flip_r  = P | (mask & (P >> dir));
    mask_l  = mask & (mask << dir);       mask_r  = mask & (mask >> dir);
    flip_l |= mask_l & (flip_l << dir2);  flip_r |= mask_r & (flip_r >> dir2);
    mask_l &= (mask_l << dir2);           mask_r &= (mask_r >> dir2);
    flip_l |= mask_l & (flip_l << dir4);  flip_r |= mask_r & (flip_r >> dir4);

    return ((flip_l & mask) << dir) | ((flip_r & mask) >> dir);

#else
  // sequential algorithm
  // 7 << + 7 >> + 6 & + 12 |
  mask_t flip;

  flip = (((P << dir) | (P >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  flip |= (((flip << dir) | (flip >> dir)) & mask);
  return (flip << dir) | (flip >> dir);

#endif
}

}  // namespace othello
