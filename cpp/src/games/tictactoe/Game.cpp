#include "games/tictactoe/Game.hpp"

namespace tictactoe {

void Game::Rules::apply(State& state, const Move& move) {
  mask_t piece_mask = mask_t(1) << int(move);
  state.cur_player_mask ^= state.full_mask;
  state.full_mask |= piece_mask;
}

Game::MoveList Game::Rules::get_legal_moves(const State& state) {
  MoveList moves;
  moves.set_all();
  uint64_t u = state.full_mask;
  while (u) {
    int index = std::countr_zero(u);
    moves.remove(index);
    u &= u - 1;
  }
  return moves;
}

void Game::IO::print_state(std::ostream& ss, const State& state, const Move* last_move,
                           const Types::player_name_array_t* player_names) {
  auto cp = Rules::get_current_player(state);
  mask_t opp_player_mask = state.opponent_mask();
  mask_t o_mask = (cp == kO) ? state.cur_player_mask : opp_player_mask;
  mask_t x_mask = (cp == kX) ? state.cur_player_mask : opp_player_mask;

  char text[] =
    "0 1 2  | | | |\n"
    "3 4 5  | | | |\n"
    "6 7 8  | | | |\n";

  int offset_table[] = {8, 10, 12, 23, 25, 27, 38, 40, 42};
  for (int i = 0; i < kNumCells; ++i) {
    int offset = offset_table[i];
    if (o_mask & (mask_t(1) << i)) {
      text[offset] = 'O';
    } else if (x_mask & (mask_t(1) << i)) {
      text[offset] = 'X';
    }
  }

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  cx += snprintf(buffer + cx, buf_size - cx, "%s\n", text);

  if (player_names) {
    cx += snprintf(buffer + cx, buf_size - cx, "X: %s\n", (*player_names)[kX].c_str());
    cx += snprintf(buffer + cx, buf_size - cx, "O: %s\n\n", (*player_names)[kO].c_str());
  }

  RELEASE_ASSERT(cx < buf_size, "Buffer overflow ({} < {})", cx, buf_size);
  ss << buffer << std::endl;
}

Game::Rules::Result Game::Rules::analyze(const State& state) {
  auto last_player = 1 - get_current_player(state);
  RELEASE_ASSERT(get_current_player(state) != last_player);  // simple sanity check

  bool win = false;

  mask_t updated_mask = state.full_mask ^ state.cur_player_mask;
  for (mask_t mask : kThreeInARowMasks) {
    if ((mask & updated_mask) == mask) {
      win = true;
      break;
    }
  }

  if (win) {
    return Result::make_terminal(GameResults::win(last_player));
  } else if (std::popcount(state.full_mask) == kNumCells) {
    return Result::make_terminal(GameResults::draw());
  }

  return Result::make_nonterminal(get_legal_moves(state));
}

}  // namespace tictactoe
