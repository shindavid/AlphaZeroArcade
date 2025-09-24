#include "games/blokus/Game.hpp"

#include "util/AnsiCodes.hpp"
#include "util/CppUtil.hpp"
#include "util/StringUtil.hpp"

#include <format>

namespace blokus {

void Game::Rules::init_state(State& state) {
  std::memset(&state, 0, sizeof(state));

  state.core.cur_color = kBlue;
  state.core.partial_move.set(-1, -1);

  for (color_t c = 0; c < kNumColors; ++c) {
    Location loc((kBoardDimension - 1) * (c % 3 > 0), (kBoardDimension - 1) * (c / 2 > 0));
    state.aux.corner_locations[c].set(loc);
  }
}

Game::Types::ActionMask Game::Rules::get_legal_moves(const State& state) {
  const State::Core& core = state.core;
  const State::Aux& aux = state.aux;

  color_t color = core.cur_color;

  Types::ActionMask valid_actions;
  if (!core.partial_move.valid()) {
    // First, we find board locations where we can fit a piece's corner

    BitBoard unplayable_locations = aux.unplayable_locations[color];
    for (Location loc : aux.corner_locations[color].get_set_locations()) {
      bool broke = false;
      corner_constraint_t constraint = aux.unplayable_locations[color].get_corner_constraint(loc);
      for (Piece piece : aux.played_pieces[color].get_unset_bits()) {
        for (PieceOrientationCorner poc : piece.get_corners(constraint)) {
          BitBoardSlice move_mask = poc.to_bitboard_mask(loc);
          if (move_mask.empty()) continue;
          if (!unplayable_locations.intersects(move_mask)) {
            valid_actions[loc.flatten()] = true;
            broke = true;
            break;
          }
        }
        if (broke) break;
      }
      // Prevent redundant representations of the same move:
      unplayable_locations.set(loc);
    }

    if (!valid_actions.any()) {
      valid_actions[kPass] = true;
    }
  } else {
    // We have a specific board location on which to place a piece's corner.
    //
    // Now we need to decide on a specific piece to play, how to orient it, and which corner of the
    // piece to place on the given location.

    Location loc = core.partial_move;

    BitBoard earlier_corner_locations = aux.corner_locations[color];
    earlier_corner_locations.clear_at_and_after(loc);  // redundancy-representation-removal
    BitBoard unplayable_locations = aux.unplayable_locations[color] | earlier_corner_locations;

    corner_constraint_t constraint = unplayable_locations.get_corner_constraint(loc);
    for (Piece piece : aux.played_pieces[color].get_unset_bits()) {
      for (PieceOrientationCorner poc : piece.get_corners(constraint)) {
        BitBoardSlice move_mask = poc.to_bitboard_mask(loc);
        if (move_mask.empty()) continue;
        if (!unplayable_locations.intersects(move_mask)) {
          valid_actions[poc.to_action()] = true;
        }
      }
    }
  }
  return valid_actions;
}

void Game::Rules::apply(State& state, core::action_t action) {
  if (IS_DEFINED(DEBUG_BUILD)) {
    state.validate_aux();
  }

  State::Core& core = state.core;
  State::Aux& aux = state.aux;

  color_t color = core.cur_color;
  if (!core.partial_move.valid()) {
    if (action == kPass) {
      core.cur_color = (core.cur_color + 1) % kNumColors;
      core.pass_count++;
    } else {
      RELEASE_ASSERT(action >= 0 && action < kPass);
      core.pass_count = 0;
      core.partial_move = Location::unflatten(action);
    }
    return;
  } else {
    Location loc = core.partial_move;
    PieceOrientationCorner poc = PieceOrientationCorner::from_action(action);
    BitBoardSlice move_mask = poc.to_bitboard_mask(loc);
    BitBoardSlice adjacent_mask = poc.to_adjacent_bitboard_mask(loc);
    BitBoardSlice diagonal_mask = poc.to_diagonal_bitboard_mask(loc);

    RELEASE_ASSERT(core.pass_count == 0);

    core.occupied_locations[color] |= move_mask;
    aux.played_pieces[color].set(poc.to_piece());
    aux.unplayable_locations[color] |= adjacent_mask;
    aux.corner_locations[color] |= diagonal_mask;

    for (color_t c = 0; c < kNumColors; ++c) {
      aux.unplayable_locations[c] |= move_mask;
      aux.corner_locations[c].unset(aux.unplayable_locations[c]);
    }

    core.cur_color = (core.cur_color + 1) % kNumColors;
    core.partial_move.set(-1, -1);
  }
}

bool Game::Rules::is_terminal(const State& state, core::seat_index_t last_player,
                              core::action_t last_action, GameResults::Tensor& outcome) {
  if (state.core.pass_count == kNumColors) {
    outcome = compute_outcome(state);
    return true;
  }
  return false;
}

Game::GameResults::Tensor Game::Rules::compute_outcome(const State& state) {
  Game::Types::ValueTensor tensor;

  int scores[kNumColors];
  for (color_t c = 0; c < kNumColors; ++c) {
    scores[c] = state.remaining_square_count(c);
  }

  int min_score = scores[0];
  for (color_t c = 1; c < kNumColors; ++c) {
    min_score = std::min(min_score, scores[c]);
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    tensor(c) = (scores[c] == min_score) ? 1 : 0;
  }

  tensor = tensor / eigen_util::sum(tensor);
  return tensor;
}

void Game::IO::print_state(std::ostream& os, const State& state, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  BoardString bs;

  for (color_t c = 0; c < kNumColors; ++c) {
    bs.set(state.core.occupied_locations[c], color_to_drawing(c));
  }

  bs.pretty_print(os);

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  static std::string color_strs[kNumColors] = {
    std::format("{}{}{}", ansi::kBlue(""), ansi::kRectangle("B"), ansi::kReset("")),
    std::format("{}{}{}", ansi::kYellow(""), ansi::kRectangle("Y"), ansi::kReset("")),
    std::format("{}{}{}", ansi::kRed(""), ansi::kRectangle("R"), ansi::kReset("")),
    std::format("{}{}{}", ansi::kGreen(""), ansi::kRectangle("G"), ansi::kReset(""))};

  cx += snprintf(buffer + cx, buf_size - cx, "\nScore: Player\n");
  for (color_t c = 0; c < kNumColors; ++c) {
    bool cur = c == state.core.cur_color;
    int score = state.remaining_square_count(c);
    cx += snprintf(buffer + cx, buf_size - cx, "%s%2d: %s", cur ? " * " : "   ", score,
                   color_strs[c].c_str());
    if (player_names) {
      cx += snprintf(buffer + cx, buf_size - cx, " [%s]", (*player_names)[c].c_str());
    }
    cx += snprintf(buffer + cx, buf_size - cx, "\n");
  }

  RELEASE_ASSERT(cx < buf_size, "Buffer overflow ({} < {})", cx, buf_size);
  os << buffer << std::endl;
}

Game::State Game::IO::load(const std::string& str, int pass_count) {
  State state;
  Rules::init_state(state);

  std::vector<std::string> lines = util::splitlines(str);
  RELEASE_ASSERT(lines.size() > 21);
  for (int row = 0; row < kBoardDimension; ++row) {
    const std::string& line = lines[20 - row];
    RELEASE_ASSERT(line.size() == 26);
    for (int col = 0; col < kBoardDimension; ++col) {
      char c = line[col + 3];
      color_t color = char_to_color(c);
      if (color == kNumColors) continue;
      state.core.occupied_locations[color].set(row, col);
    }
  }

  state.compute_aux();
  state.core.pass_count = pass_count;
  state.core.cur_color = (state.last_placed_piece_color() + pass_count + 1) % kNumColors;

  return state;
}

std::string Game::IO::player_to_str(core::seat_index_t player) {
  switch (player) {
    case blokus::kBlue:
      return std::format("{}{}{}", ansi::kBlue(""), ansi::kRectangle("B"), ansi::kReset(""));
    case blokus::kYellow:
      return std::format("{}{}{}", ansi::kYellow(""), ansi::kRectangle("Y"), ansi::kReset(""));
    case blokus::kRed:
      return std::format("{}{}{}", ansi::kRed(""), ansi::kRectangle("R"), ansi::kReset(""));
    case blokus::kGreen:
      return std::format("{}{}{}", ansi::kGreen(""), ansi::kRectangle("G"), ansi::kReset(""));
    default:
      return "?";
  }
}

}  // namespace blokus
