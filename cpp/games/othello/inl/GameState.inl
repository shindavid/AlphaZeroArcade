#include <games/othello/GameState.hpp>

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

// copied from edax-reversi repo - board_next()
inline core::GameStateTypes<GameState>::GameOutcome GameState::apply_move(core::action_index_t action) {
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

template<eigen_util::FixedTensorConcept InputTensor> void GameState::tensorize(InputTensor& tensor) const {
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      int index = row * kBoardDimension + col;
      bool occupied_by_cur_player = (1UL << index) & cur_player_mask_;
      tensor(0, row, col) = occupied_by_cur_player;
    }
  }
  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      int index = row * kBoardDimension + col;
      bool occupied_by_opp_player = (1UL << index) & opponent_mask_;
      tensor(1, row, col) = occupied_by_opp_player;
    }
  }
}

inline void GameState::dump(core::action_index_t last_action, const player_name_array_t* player_names) const {
  ActionMask valid_actions = get_valid_actions();
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
  printf("   A B C D E F G H\n");
  for (int row = 0; row < kBoardDimension; ++row) {
    row_dump(valid_actions, row, row == blink_row ? blink_col : -1);
  }
  std::cout << std::endl;
  int opponent_disc_count = std::popcount(opponent_mask_);
  int cur_player_disc_count = std::popcount(cur_player_mask_);

  int black_disc_count = cur_player_ == kBlack ? cur_player_disc_count : opponent_disc_count;
  int white_disc_count = cur_player_ == kWhite ? cur_player_disc_count : opponent_disc_count;

  printf("Score: Player\n");
  printf("%5d: %s%s%s", black_disc_count, ansi::kBlue(""), ansi::kCircle("*"), ansi::kReset(""));
  if (player_names) {
    printf(" [%s]", (*player_names)[kBlack].c_str());
  }
  printf("\n");

  printf("%5d: %s%s%s", white_disc_count, ansi::kWhite(""), ansi::kCircle("0"), ansi::kReset(""));
  if (player_names) {
    printf(" [%s]", (*player_names)[kWhite].c_str());
  }
  printf("\n");
  std::cout << std::endl;
}

inline std::size_t GameState::hash() const {
  return util::tuple_hash(to_tuple());
}

inline void GameState::row_dump(const ActionMask& valid_actions, row_t row, column_t blink_column) const {
  core::seat_index_t current_player = get_current_player();
  const char* cur_color = current_player == kBlack ? ansi::kBlue("*") : ansi::kWhite("0");
  const char* opp_color = current_player == kBlack ? ansi::kWhite("0") : ansi::kBlue("*");

  char prefix = ' ';
  if (!util::tty_mode() && blink_column >= 0) {
    prefix = 'x';
  }
  printf("%c%d", prefix, (int)(row+1));
  for (int col = 0; col < kBoardDimension; ++col) {
    int index = row * kBoardDimension + col;
    bool valid = valid_actions[index];
    bool occupied_by_cur_player = (1UL << index) & cur_player_mask_;
    bool occupied_by_opp_player = (1UL << index) & opponent_mask_;
    bool occupied = occupied_by_cur_player || occupied_by_opp_player;

    const char* color = occupied_by_cur_player ? cur_color : (occupied_by_opp_player ? opp_color : "");
    const char* c = occupied ? ansi::kCircle("") : (valid ? "." : " ");

    printf("|%s%s%s%s", col == blink_column ? ansi::kBlink("") : "", color, c, occupied ? ansi::kReset("") : "");
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

namespace core {

inline void MctsResultsDumper<othello::GameState>::dump(
    const LocalPolicyArray &action_policy, const MctsResults &results)
{
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  assert(net_policy.size() == (int)valid_actions.count());

  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kBlue(""), ansi::kCircle("*"), ansi::kReset(""),
         100 * net_value(othello::kBlack), 100 * win_rates(othello::kBlack));
  printf("%s%s%s: %6.3f%% -> %6.3f%%\n", ansi::kWhite(""), ansi::kCircle("0"), ansi::kReset(""),
         100 * net_value(othello::kWhite), 100 * win_rates(othello::kWhite));
  printf("\n");

  auto tuple0 = std::make_tuple(mcts_counts(0), action_policy(0), net_policy(0), 0);
  using tuple_t = decltype(tuple0);
  using tuple_array_t = std::array<tuple_t, othello::kNumGlobalActions>;
  tuple_array_t tuples;
  int i = 0;
  for (core::action_index_t action : bitset_util::on_indices(valid_actions)) {
    tuples[i] = std::make_tuple(mcts_counts.data()[action], action_policy(i), net_policy(i), action);
    i++;
  }

  std::sort(tuples.begin(), tuples.end());
  std::reverse(tuples.begin(), tuples.end());

  int num_rows = 10;
  int num_actions = i;
  printf("%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  for (i = 0; i < std::min(num_rows, num_actions); ++i) {
    const auto& tuple = tuples[i];

    float count = std::get<0>(tuple);
    auto action_p = std::get<1>(tuple);
    auto net_p = std::get<2>(tuple);
    int action = std::get<3>(tuple);

    if (action == othello::kPass) {
      printf("%4s %8.3f %8.3f %8.3f\n", "Pass", net_p, count, action_p);
    } else {
      int row = action / othello::kBoardDimension;
      int col = action % othello::kBoardDimension;
      printf("  %c%d %8.3f %8.3f %8.3f\n", 'A' + col, row + 1, net_p, count, action_p);
    }
  }
  for (i = num_actions; i < num_rows; ++i) {
    printf("\n");
  }
}


}  // namespace core
