#include <games/blokus/Game.hpp>

#include <util/CppUtil.hpp>

namespace blokus {

void Game::Rules::init_state(FullState& state, group::element_t sym) {
  util::release_assert(sym == group::kIdentity);
  std::memset(&state, 0, sizeof(state));

  FullState::core_t& core = state.core;
  FullState::aux_t& aux = state.aux;

  core.cur_color = kBlue;
  core.partial_move.set(-1, -1);

  constexpr int B = kBoardDimension;

  aux.corner_locations[kBlue].set(0, 0);
  aux.corner_locations[kYellow].set(B - 1, 0);
  aux.corner_locations[kRed].set(B - 1, B - 1);
  aux.corner_locations[kGreen].set(0, B - 1);
}

Game::Types::ActionMask Game::Rules::get_legal_moves(const FullState& state) {
  const FullState::core_t& core = state.core;
  const FullState::aux_t& aux = state.aux;

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
            valid_actions[loc] = true;
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

    corner_constraint_t constraint = unplayable_locations[color].get_corner_constraint(loc);
    for (Piece piece : aux.played_pieces[color].get_unset_bits()) {
      for (PieceOrientationCorner poc : piece.get_corners(constraint)) {
        BitBoardSlice move_mask = poc.to_bitboard_mask(loc);
        if (move_mask.empty()) continue;
        if (!unplayable_locations.intersects(move_mask)) {
          valid_actions[s] = true;
        }
      }
    }
  }
  return valid_actions;
}

Game::Types::ActionOutcome Game::Rules::apply(FullState& state, core::action_t action) {
  FullState::core_t& core = state.core;
  FullState::aux_t& aux = state.aux;

  color_t color = core.cur_color;
  if (!core.partial_move.valid()) {
    if (action == kPass) {
      if (core.pass_count == 3) {  // all players passed, game over
        return compute_outcome(state);
      }
      core.cur_color = (core.cur_color + 1) % kNumColors;
      core.pass_count++;
      return Types::ActionOutcome();
    } else {
      util::release_assert(action >= 0 && action < kPass);
      core.pass_count = 0;
      core.partial_move = action;
      return Types::ActionOutcome();
    }
  } else {
    Location loc = core.partial_move;
    PieceOrientationCorner poc(action);
    BitBoardSlice move_mask = poc.to_bitboard_mask(loc);
    BitBoardSlice adjacent_mask = poc.to_adjacent_bitboard_mask(loc);
    BitBoardSlice diagonal_mask = poc.to_diagonal_bitboard_mask(loc);

    util::release_assert(core.pass_count == 0);

    core.occupied_locations[color] |= move_mask;
    aux.played_pieces[color].set(poc.piece_index());
    aux.unplayable_locations[color] |= adjacent_mask;
    aux.corner_locations[color] |= diagonal_mask;

    for (color_t c = 0; c < kNumColors; ++c) {
      aux.unplayable_locations[c] |= move_mask;
      aux.corner_locations[c] &= ~aux.unplayable_locations[c];
    }

    core.cur_color = (core.cur_color + 1) % kNumColors;
    core.partial_move.set(-1, -1);
    return Types::ActionOutcome();
  }
}

Game::Types::ActionOutcome Game::Rules::compute_outcome(const FullState& state) {
  const FullState::core_t& core = state.core;
  Game::Types::ValueArray array;

  int scores[kNumColors];
  for (color_t c = 0; c < kNumColors; ++c) {
    scores[c] = kNumCells - core.occupied_locations[c].count();
  }

  int min_score = scores[0];
  for (color_t c = 1; c < kNumColors; ++c) {
    min_score = std::min(min_score, scores[c]);
  }

  for (color_t c = 0; c < kNumColors; ++c) {
    array[c] = (scores[c] == min_score) ? 1 : 0;
  }

  array /= array.sum();
  return array;
}

void Game::IO::print_state(std::ostream&, const BaseState&, core::action_t last_action,
                           const Types::player_name_array_t* player_names) {
  throw std::runtime_error("Not implemented");
}

void Game::IO::print_mcts_results(std::ostream&, const Types::PolicyTensor& action_policy,
                                  const Types::SearchResults&) {
  throw std::runtime_error("Not implemented");
}

Game::InputTensorizor::Tensor Game::InputTensorizor::tensorize(const BaseState* start,
                                                               const BaseState* cur) {
  throw std::runtime_error("Not implemented");
}

}  // namespace blokus
