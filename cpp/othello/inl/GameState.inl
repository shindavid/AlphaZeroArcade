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
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      int index = row * kBoardDimension + col;
      bool occupied_by_cur_player = (1UL << index) & cur_player_mask_;
      tensor(0, 0, row, col) = occupied_by_cur_player;
    }
  }
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      int index = row * kBoardDimension + col;
      bool occupied_by_opp_player = (1UL << index) & opponent_mask_;
      tensor(0, 1, row, col) = occupied_by_opp_player;
    }
  }
}

inline void GameState::dump(common::action_index_t last_action, const player_name_array_t* player_names) const {
  bool display_last_action = last_action >= 0;
  int blink_row = -1;
  int blink_col = -1;
  if (display_last_action && last_action != kPass) {
    blink_row = last_action / kBoardDimension;
    blink_col = last_action % kBoardDimension;
  }
  if (!util::tty_mode() && display_last_action) {
    std::string s(2*blink_col+3, ' ');
    printf("%sx\n", s.c_str());
  }
  printf("  |A|B|C|D|E|F|G|H|\n");
  for (int row = 0; row < kBoardDimension; ++row) {
    row_dump(row, row == blink_row ? blink_col : -1);
  }
  if (player_names) {
    printf("%s%s%s: %s\n", ansi::kBlue(), ansi::kCircle(), ansi::kReset(), (*player_names)[kBlack].c_str());
    printf("%s%s%s: %s\n\n", ansi::kWhite(), ansi::kCircle(), ansi::kReset(), (*player_names)[kWhite].c_str());
  }
  std::cout.flush();
}

inline std::size_t GameState::hash() const {
  return util::tuple_hash(to_tuple());
}

inline void GameState::dump_mcts_output(
    const ValueProbDistr& mcts_value, const LocalPolicyProbDistr& mcts_policy, const MctsResults& results)
{
  const auto& valid_actions = results.valid_actions;
  const auto& net_value = results.value_prior;
  const auto& net_policy = results.policy_prior;
  const auto& mcts_counts = results.counts;

  assert(net_policy.size() == (int)valid_actions.count());

  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kBlue(), ansi::kCircle(), ansi::kReset(), 100 * net_value(kBlack),
         100 * mcts_value(kBlack));
  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kWhite(), ansi::kCircle(), ansi::kReset(), 100 * net_value(kWhite),
         100 * mcts_value(kWhite));
  printf("\n");

  auto tuple0 = std::make_tuple(valid_actions[0], mcts_counts(0), mcts_policy(0), net_policy(0), 0);
  using tuple_t = decltype(tuple0);
  using tuple_array_t = std::array<tuple_t, kNumGlobalActions>;
  tuple_array_t tuples;
  tuples[0] = tuple0;
  for (int i = 1; i < kNumGlobalActions; ++i) {
    tuples[i] = std::make_tuple(valid_actions[i], mcts_counts(i), mcts_policy(i), net_policy(i), i);
  }
  std::sort(tuples.begin(), tuples.end());
  std::reverse(tuples.begin(), tuples.end());

  constexpr int kNumActionsToPrint = 10;
  printf("%4s %8s %8s %8s\n", "Col", "Net", "Count", "MCTS");
  for (int i = 0; i < kNumActionsToPrint; ++i) {
    const auto& tuple = tuples[i];
    bool valid_action = std::get<0>(tuple);
    if (!valid_action) {
      printf("\n");
      continue;
    }

    float count = std::get<1>(tuple);
    auto mcts_p = std::get<2>(tuple);
    auto net_p = std::get<3>(tuple);
    int action = std::get<4>(tuple);

    if (action == kPass) {
      printf("%4s %8.3f %8.3f %8.3f\n", "Pass", net_p, count, mcts_p);
    } else {
      int row = action / kBoardDimension;
      int col = action % kBoardDimension;
      printf("  %c%d %8.3f %8.3f %8.3f\n", 'A' + row, col + 1, net_p, count, mcts_p);
    }
  }
}

inline void GameState::row_dump(row_t row, column_t blink_column) const {
  common::seat_index_t current_player = get_current_player();
  const char* cur_color = current_player == kBlack ? ansi::kBlue() : ansi::kWhite();
  const char* opp_color = current_player == kBlack ? ansi::kWhite() : ansi::kBlue();

  char prefix = ' ';
  if (!util::tty_mode() && blink_column >= 0) {
    prefix = 'x';
  }
  printf("%c%d", prefix, (int)(row+1));
  for (int col = 0; col < kBoardDimension; ++col) {
    int index = row * kBoardDimension + col;
    bool occupied_by_cur_player = (1UL << index) & cur_player_mask_;
    bool occupied_by_opp_player = (1UL << index) & opponent_mask_;
    bool occupied = occupied_by_cur_player || occupied_by_opp_player;

    const char* color = occupied_by_cur_player ? cur_color : (occupied_by_opp_player ? opp_color : "");
    const char* c = occupied ? ansi::kCircle() : " ";

    printf("|%s%s%s%s", col == blink_column ? ansi::kBlink() : "", color, c, occupied ? ansi::kReset() : "");
  }

  printf("|\n");
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
