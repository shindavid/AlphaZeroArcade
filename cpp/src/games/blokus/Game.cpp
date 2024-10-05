#include <games/blokus/Game.hpp>

#include <util/AnsiCodes.hpp>
#include <util/CppUtil.hpp>
#include <util/StringUtil.hpp>

namespace blokus {

color_t Game::State::last_placed_piece_color() const {
  int max = -1;
  color_t last_color = kNumColors;
  for (color_t c = 0; c < kNumColors; ++c) {
    int count = aux.played_pieces[c].count();
    if (count > max) {
      max = count;
      last_color = c;
    }
  }
  return last_color;
}

void Game::State::compute_aux() {
  BitBoard occupied;
  occupied.clear();

  for (color_t c = 0; c < kNumColors; ++c) {
    occupied |= core.occupied_locations[c];
    aux.corner_locations[c].clear();
    if (!core.occupied_locations[c].any()) {
      Location loc((kBoardDimension - 1) * (c % 3 > 0), (kBoardDimension - 1) * (c / 2 > 0));
      aux.corner_locations[c].set(loc);
    }
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    aux.unplayable_locations[c] = occupied | core.occupied_locations[c].adjacent_squares();
    aux.corner_locations[c] |= core.occupied_locations[c].diagonal_squares();
    aux.corner_locations[c].unset(aux.unplayable_locations[c]);
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    aux.played_pieces[c].clear();

    occupied = core.occupied_locations[c];
    for (Location loc : occupied.get_set_locations()) {
      PieceOrientationCorner poc = occupied.find(loc);
      occupied.unset(poc.to_bitboard_mask(loc));
      aux.played_pieces[c].set(poc.to_piece());
    }
  }
}

void Game::State::validate_aux() const {
  State copy = *this;
  copy.compute_aux();

  if (copy.aux != this->aux) {
    printf("blokus::Game::State validation failure!\n\n");
    std::ostringstream ss;
    Game::IO::print_state(ss, *this, kPass+1);
    std::cout << ss.str() << std::endl;

    for (color_t c = 0; c < kNumColors; ++c) {
      PieceMask diff1 = copy.aux.played_pieces[c] & ~this->aux.played_pieces[c];
      PieceMask diff2 = this->aux.played_pieces[c] & ~copy.aux.played_pieces[c];
      for (Piece p : diff1.get_set_bits()) {
        printf("played_pieces %c %d: this=0 copy=1\n", color_to_char(c), (int)p);
      }
      for (Piece p : diff2.get_set_bits()) {
        printf("played_pieces %c %d: this=1 copy=0\n", color_to_char(c), (int)p);
      }

      BitBoard diff3 = copy.aux.unplayable_locations[c] & ~this->aux.unplayable_locations[c];
      BitBoard diff4 = this->aux.unplayable_locations[c] & ~copy.aux.unplayable_locations[c];
      for (Location loc : diff3.get_set_locations()) {
        printf("unplayable_locations %c@%s: this=0 copy=1\n", color_to_char(c),
               loc.to_string().c_str());
      }
      for (Location loc : diff4.get_set_locations()) {
        printf("unplayable_locations %c@%s: this=1 copy=0\n", color_to_char(c),
               loc.to_string().c_str());
      }

      BitBoard diff5 = copy.aux.corner_locations[c] & ~this->aux.corner_locations[c];
      BitBoard diff6 = this->aux.corner_locations[c] & ~copy.aux.corner_locations[c];
      for (Location loc : diff5.get_set_locations()) {
        printf("corner_locations %c@%s: this=0 copy=1\n", color_to_char(c),
               loc.to_string().c_str());
      }
      for (Location loc : diff6.get_set_locations()) {
        printf("corner_locations %c@%s: this=1 copy=0\n", color_to_char(c),
               loc.to_string().c_str());
      }
    }
    throw util::Exception("Auxiliary data is inconsistent with core data");
  }
}

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

Game::Types::ActionOutcome Game::Rules::apply(StateHistory& history, core::action_t action) {
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
      if (core.pass_count == 4) {  // all players passed, game over
        return compute_outcome(state);
      }
      return Types::ActionOutcome();
    } else {
      util::release_assert(action >= 0 && action < kPass);
      core.pass_count = 0;
      core.partial_move = Location::unflatten(action);
      return Types::ActionOutcome();
    }
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
    return Types::ActionOutcome();
  }
}

Game::Types::ActionOutcome Game::Rules::compute_outcome(const State& state) {
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

void Game::IO::print_mcts_results(std::ostream& os, const Types::PolicyTensor& action_policy,
                                  const Types::SearchResults& results) {
  const auto& valid_actions = results.valid_actions;
  const auto& mcts_counts = results.counts;
  const auto& net_policy = results.policy_prior;
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

  auto tuple0 = std::make_tuple(mcts_counts(0), action_policy(0), net_policy(0), 0);
  using tuple_t = decltype(tuple0);
  using tuple_array_t = std::array<tuple_t, kNumActions>;
  tuple_array_t tuples;
  int i = 0;
  for (int a = 0; a < kNumActions; ++a) {
    if (valid_actions[a]) {
      tuples[i] = std::make_tuple(mcts_counts(a), action_policy(a), net_policy(a), a);
      i++;
    }
  }

  std::sort(tuples.begin(), tuples.end());
  std::reverse(tuples.begin(), tuples.end());

  int num_rows = 10;
  int num_actions = i;
  cx += snprintf(buffer + cx, buf_size - cx, "%4s %8s %8s %8s\n", "Move", "Net", "Count", "MCTS");
  for (i = 0; i < std::min(num_rows, num_actions); ++i) {
    const auto& tuple = tuples[i];

    float count = std::get<0>(tuple);
    auto action_p = std::get<1>(tuple);
    auto net_p = std::get<2>(tuple);
    int action = std::get<3>(tuple);

    std::string action_str;
    if (action < kPass) {  // location
      action_str = Location::unflatten(action).to_string();
    } else if (action > kPass) {
      action_str = PieceOrientationCorner::from_action(action).name();
    } else {
      action_str = "Pass";
    }
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
