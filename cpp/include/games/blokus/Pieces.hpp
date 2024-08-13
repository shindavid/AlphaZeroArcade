#pragma once

#include <core/BasicTypes.hpp>
#include <util/FiniteGroups.hpp>

#include <bit>
#include <cstdint>

namespace blokus {

struct PieceOrientation {
  PieceOrientation(piece_orientation_index_t, piece_index_t, group::element_t, column_mask_t c0,
                   column_mask_t c1 = 0, column_mask_t c2 = 0, column_mask_t c3 = 0,
                   column_mask_t c4 = 0);
  PieceOrientation(const PieceOrientation&) = delete;
  PieceOrientation& operator=(const PieceOrientation&) = delete;

  piece_orientation_index_t index;
  piece_index_t piece_index;
  group::element_t orientation;
  column_mask_t column_masks[5];
};

extern const PieceOrientation kPieceOrientations[kNumPieceOrientations];

struct Piece {
  Piece(const char*, piece_index_t, piece_orientation_index_t canonical_poi,
        group_subset_t, column_mask_t c0, column_mask_t c1 = 0, column_mask_t c2 = 0);
  Piece(const Piece&) = delete;
  Piece& operator=(const Piece&) = delete;

  oriented_piece_index_t orient(group::element_t sym) const;
  int num_orientations() const { return std::popcount(orientations); }

  char name[3];
  piece_index_t index;
  piece_orientation_index_t canonical_orientation_index;
  group_subset_t orientations;
  column_mask_t column_masks[3];
};

const piece_index_t pO1 = 0;
const piece_index_t pI2 = 1;
const piece_index_t pI3 = 2;
const piece_index_t pL3 = 3;
const piece_index_t pI4 = 4;
const piece_index_t pO4 = 5;
const piece_index_t pT4 = 6;
const piece_index_t pL4 = 7;
const piece_index_t pS4 = 8;
const piece_index_t pF5 = 9;
const piece_index_t pI5 = 10;
const piece_index_t pL5 = 11;
const piece_index_t pN5 = 12;
const piece_index_t pP5 = 13;
const piece_index_t pT5 = 14;
const piece_index_t pU5 = 15;
const piece_index_t pV5 = 16;
const piece_index_t pW5 = 17;
const piece_index_t pX5 = 18;
const piece_index_t pY5 = 19;
const piece_index_t pZ5 = 20;
const piece_index_t kNumPieces = 21;

extern const Piece kPieces[kNumPieces];

}  // namespace blokus

#include <inline/games/blokus/Pieces.inl>
