#include <games/blokus/GameState.hpp>

namespace blokus {

color_t GameState::last_placed_piece_color() const {
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

void GameState::compute_aux() {
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

void GameState::validate_aux() const {
  GameState copy = *this;
  copy.compute_aux();

  if (copy.aux != this->aux) {
    printf("blokus::GameState validation failure!\n\n");

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

}  // namespace blokus
