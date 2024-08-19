#pragma once

#include <core/BasicTypes.hpp>
#include <util/FiniteGroups.hpp>

#include <bit>
#include <cstdint>

namespace blokus {

using direction_t = uint8_t;

const direction_t dN = 0;
const direction_t dE = 0;
const direction_t dS = 0;
const direction_t dW = 0;

#pragma pack(push, 1)
class CornerConstraint {
 public:
  void set(direction_t dir, bool value);
  int get_count() const;  // number of directions that are constrained

 private:
  uint8_t mask_ = 0;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Location {
  void set(int8_t row, int8_t col);
  bool valid() const;

  int8_t row;
  int8_t col;
};
#pragma pack(pop)
static_assert(sizeof(Location) == 2);

#pragma pack(push, 1)
class PieceOrientationCorner {
 public:
  PieceOrientationCorner(piece_orientation_corner_index_t index) : index_(index) {}
  piece_orientation_corner_index_t operator() const { return index_; }
  piece_index_t piece_index() const;

 private:
  piece_orientation_corner_index_t index_;
};
#pragma pack(pop)
static_assert(sizeof(PieceOrientationCorner) == 2);

#pragma pack(push, 1)
class PieceOrientation {
 public:
  PieceOrientation(piece_orientation_index_t index) : index_(index) {}
  piece_orientation_index_t operator() const { return index_; }

 private:
  piece_orientation_index_t index_;
};
#pragma pack(pop)
static_assert(sizeof(PieceOrientation) == 1);

#pragma pack(push, 1)
class Piece {
 public:
  Piece(piece_index_t index) : index_(index) {}
  piece_index_t operator() const { return index_; }
  const char* name() const;
  auto orientations() const;
  auto get_corners(CornerConstraint constraint) const;

 private:
  piece_index_t index_;
};
#pragma pack(pop)
static_assert(sizeof(Piece) == 1);

// extern const PieceOrientationSquare kPieceOrientationSquares[kNumPieceOrientationSquares];

/*
 * piece_mask_t represents a subset of squares of a 5x5 grid occupied by an oriented piece that
 * fits snugly against the bottom and left edges of the grid. They are numbered as follows:
 *
 * 14
 * 12 13
 *  9 10 11
 *  5  6  7  8
 *  0  1  2  3  4
 */
// #pragma pack(push, 1)
// struct piece_mask_t {
//   uint16_t mask;
// };
// #pragma pack(pop)

// struct PieceOrientation {
//   PieceOrientation(piece_index_t, column_mask_t c0,
//                    column_mask_t c1 = 0, column_mask_t c2 = 0, column_mask_t c3 = 0,
//                    column_mask_t c4 = 0);
//   PieceOrientation(const PieceOrientation&) = delete;
//   PieceOrientation& operator=(const PieceOrientation&) = delete;

//   piece_mask_t occupied_squares;
//   piece_mask_t diagonal_squares[kNumDiagonalDirections];
//   int8_t width;
//   int8_t height;
//   piece_index_t piece_index;
// };
// static_assert(sizeof(PieceOrientation) == 16);

// extern const PieceOrientation kPieceOrientations[kNumPieceOrientations];

// struct Piece {
//   Piece(const char*, piece_orientation_index_t canonical_poi,
//         group_subset_t, column_mask_t c0, column_mask_t c1 = 0, column_mask_t c2 = 0);
//   Piece(const Piece&) = delete;
//   Piece& operator=(const Piece&) = delete;

//   oriented_piece_index_t orient(group::element_t sym) const;
//   bool accepts(group::element_t sym) const { return orientations & (1 << sym); }
//   int num_orientations() const { return std::popcount(orientations); }

//   char name[3];
//   piece_orientation_index_t canonical_orientation_index;
//   group_subset_t orientations;
//   piece_mask_t occupied_squares;
// };
// static_assert(sizeof(Piece) == 8);

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

// extern const Piece kPieces[kNumPieces];

}  // namespace blokus

#include <inline/games/blokus/Pieces.inl>
