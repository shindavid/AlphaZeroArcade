#include <games/blokus/Game.hpp>

#include <util/AnsiCodes.hpp>
#include <util/CppUtil.hpp>
#include <util/StringUtil.hpp>

namespace blokus {

void Game::Rules::init_state(State& state) {
  std::memset(&state, 0, sizeof(state));

  state.core.cur_color = kBlue;
  state.core.partial_move.set(-1, -1);

  for (color_t c = 0; c < kNumColors; ++c) {
    Location loc((kBoardDimension-1)*(c%3>0), (kBoardDimension-1)*(c/2>0));
    state.aux.corner_locations[c].set(loc);
  }
}

Game::Types::ActionMask Game::Rules::get_legal_moves(const StateHistory& history) {
  const State& state = history.current();
  const State::core_t& core = state.core;
  const State::aux_t& aux = state.aux;

  color_t color = core.cur_color;

  using Bitset = mp::TypeAt_t<Types::ActionMask, 0>;
  Bitset valid_actions;
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

void Game::Rules::apply(StateHistory& history, core::action_t action) {
  State& state = history.extend();

  if (IS_MACRO_ENABLED(DEBUG_BUILD)) {
    state.validate_aux();
  }

  State::core_t& core = state.core;
  State::aux_t& aux = state.aux;

  color_t color = core.cur_color;
  if (!core.partial_move.valid()) {
    if (action == kPass) {
      core.cur_color = (core.cur_color + 1) % kNumColors;
      core.pass_count++;
    } else {
      util::release_assert(action >= 0 && action < kPass);
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

    util::release_assert(core.pass_count == 0);

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
      util::create_string("%s%s%s", ansi::kBlue(""), ansi::kRectangle("B"), ansi::kReset("")),
      util::create_string("%s%s%s", ansi::kYellow(""), ansi::kRectangle("Y"), ansi::kReset("")),
      util::create_string("%s%s%s", ansi::kRed(""), ansi::kRectangle("R"), ansi::kReset("")),
      util::create_string("%s%s%s", ansi::kGreen(""), ansi::kRectangle("G"), ansi::kReset(""))};

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

  util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
  os << buffer << std::endl;
}

void Game::IO::print_mcts_results(std::ostream& os, const Types::Policy& action_policy,
                                  const Types::SearchResults& results) {
  const auto& valid_actions = std::get<0>(results.valid_actions);
  const auto& mcts_counts = std::get<0>(results.counts);
  const auto& net_policy = std::get<0>(results.policy_prior);
  const auto& action_subpolicy = std::get<0>(action_policy);
  const auto& win_rates = results.win_rates;
  const auto& net_value = results.value_prior;

  constexpr int buf_size = 4096;
  char buffer[buf_size];
  int cx = 0;

  static std::string color_strs[kNumColors] = {
      util::create_string("%s%s%s", ansi::kBlue(""), ansi::kRectangle("B"), ansi::kReset("")),
      util::create_string("%s%s%s", ansi::kYellow(""), ansi::kRectangle("Y"), ansi::kReset("")),
      util::create_string("%s%s%s", ansi::kRed(""), ansi::kRectangle("R"), ansi::kReset("")),
      util::create_string("%s%s%s", ansi::kGreen(""), ansi::kRectangle("G"), ansi::kReset(""))};

  for (color_t c = 0; c < kNumColors; ++c) {
    cx += snprintf(buffer + cx, buf_size - cx, "%s: %6.3f%% -> %6.3f%%\n", color_strs[c].c_str(),
                   100 * net_value(c), 100 * win_rates(c));
  }
  cx += snprintf(buffer + cx, buf_size - cx, "\n");

  auto tuple0 = std::make_tuple(mcts_counts(0), action_subpolicy(0), net_policy(0), 0);
  using tuple_t = decltype(tuple0);
  using tuple_array_t = std::array<tuple_t, kNumActions>;
  tuple_array_t tuples;
  int i = 0;
  for (int a = 0; a < kNumActions; ++a) {
    if (valid_actions[a]) {
      tuples[i] = std::make_tuple(mcts_counts(a), action_subpolicy(a), net_policy(a), a);
      i++;
    }
  }
  int num_actions = i;

  std::sort(tuples.begin(), tuples.begin() + num_actions);
  std::reverse(tuples.begin(), tuples.begin() + num_actions);

  int num_rows = 10;
  cx += snprintf(buffer + cx, buf_size - cx, "%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  for (i = 0; i < std::min(num_rows, num_actions); ++i) {
    const auto& tuple = tuples[i];

    float count = std::get<0>(tuple);
    auto action_p = std::get<1>(tuple);
    auto net_p = std::get<2>(tuple);
    int action = std::get<3>(tuple);

    std::string action_str = action_to_str(action);
    cx += snprintf(buffer + cx, buf_size - cx, "%4s %8.3f %8.3f %8.3f\n", action_str.c_str(), net_p,
                   count, action_p);
  }
  for (i = num_actions; i < num_rows; ++i) {
    cx += snprintf(buffer + cx, buf_size - cx, "\n");
  }

  util::release_assert(cx < buf_size, "Buffer overflow (%d < %d)", cx, buf_size);
  os << buffer << std::endl;
}

Game::State Game::IO::load(const std::string& str, int pass_count) {
  State state;
  Rules::init_state(state);

  std::vector<std::string> lines = util::splitlines(str);
  util::release_assert(lines.size() > 21);
  for (int row = 0; row < kBoardDimension; ++row) {
    const std::string& line = lines[20 - row];
    util::release_assert(line.size() == 26);
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

Game::TrainingTargets::ScoreTarget::Tensor
Game::TrainingTargets::ScoreTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  tensor.setZero();
  const State& state = *view.final_pos;
  color_t cp = Rules::get_current_player(*view.cur_pos);

  int scores[kNumColors];
  for (color_t c = 0; c < kNumColors; ++c) {
    int score = state.remaining_square_count(c);
    scores[c] = std::min(score, kVeryBadScore);
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    color_t rc = (kNumColors + c - cp) % kNumColors;

    // PDF
    tensor(0, scores[c], rc) = 1;

    // CDF
    for (int score = 0; score <= scores[c]; ++score) {
      tensor(1, score, rc) = 1;
    }
  }

  return tensor;
}

Game::TrainingTargets::OwnershipTarget::Tensor Game::TrainingTargets::OwnershipTarget::tensorize(
    const Types::GameLogView& view) {
  Tensor tensor;
  tensor.setZero();
  const State& state = *view.final_pos;
  color_t cp = Rules::get_current_player(*view.cur_pos);

  for (int row = 0; row < kBoardDimension; ++row) {
    for (int col = 0; col < kBoardDimension; ++col) {
      tensor(kNumColors, row, col) = 1;
    }
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    color_t rc = (kNumColors + c - cp) % kNumColors;
    for (Location loc : state.core.occupied_locations[c].get_set_locations()) {
      tensor(rc, loc.row, loc.col) = 1;
      tensor(kNumColors, loc.row, loc.col) = 0;
    }
  }

  return tensor;
}

Game::TrainingTargets::UnplayedPiecesTarget::Tensor
Game::TrainingTargets::UnplayedPiecesTarget::tensorize(const Types::GameLogView& view) {
  Tensor tensor;
  tensor.setZero();
  const State& state = *view.final_pos;
  color_t cp = Rules::get_current_player(*view.cur_pos);

  for (color_t c = 0; c < kNumColors; ++c) {
    const PieceMask& mask = state.aux.played_pieces[c];
    color_t rc = (kNumColors + c - cp) % kNumColors;
    for (auto p : mask.get_unset_bits()) {
      tensor(rc, p) = 1;
    }
  }

  return tensor;
}

}  // namespace blokus
