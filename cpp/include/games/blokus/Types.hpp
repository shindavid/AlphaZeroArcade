#pragma once

#include <core/BasicTypes.hpp>
#include <games/blokus/Constants.hpp>
#include <util/FiniteGroups.hpp>

#include <bit>
#include <bitset>
#include <cstdint>

/*
 * Most of the core Blokus types are simply wrappers around a single integer value. This integer is
 * an index into a table of precomputed values.
 */
namespace blokus {

#pragma pack(push, 1)
struct Location {
  void set(int8_t row, int8_t col);
  bool valid() const;

  int8_t row;
  int8_t col;
};
#pragma pack(pop)
static_assert(sizeof(Location) == 2);

// Backs the Piece type
struct _PieceData {
  std::string name;
  int corner_array_start_index;
  int num_corner_array_indices;
};

// Backs the PieceOrientation type
#pragma pack(push, 1)
struct _PieceOrientationData {
  int16_t mask_array_start_index;
  int8_t height;
  int8_t width;
};
#pragma pack(pop)
static_assert(sizeof(_PieceOrientationData) == 4);

// Backs the PieceOrientationCorner type
#pragma pack(push, 1)
struct _PieceOrientationCornerData {
  Location corner_offset;
  piece_index_t piece;
  piece_orientation_index_t piece_orientation;
};
#pragma pack(pop)
static_assert(sizeof(_PieceOrientationCornerData) == 4);

class BitBoard;
class BitBoardSlice;

class BoardString {
 public:
  BoardString();
  void print(std::ostream&) const;
  void set(Location loc, const std::string& str) { strs_[loc.row][loc.col] = str; }
  void set(const BitBoard& board, const std::string& str);
  void set(const BitBoardSlice& board, const std::string& str);

 private:
  std::string strs_[kBoardDimension][kBoardDimension];
};

/*
 * BitBoard is suitable for storing a board-mask for the entire board
 */
class BitBoard {
 public:
  BitBoard operator|(const BitBoard& other) const;
  BitBoard& operator&=(const BitBoard& other);
  BitBoard operator~() const;
  BitBoard& operator|=(const BitBoardSlice& other);

  bool any() const;
  void clear();
  int count();
  void clear_at_and_after(const Location& loc);
  uint32_t get_row(int k) const { return rows[k]; }
  bool get(int row, int col) const;
  void set(int row, int col);
  void set(const Location& loc);
  auto get_set_locations() const;
  void write_to(std::bitset<kNumCells>& bitset) const;
  corner_constraint_t get_corner_constraint(Location loc) const;
  bool intersects(const BitBoardSlice& other) const;

 protected:
  uint32_t rows[kBoardDimension];
};

/*
 * BitBoardSlice uses only a subset of the rows of the board. The memory corresponding to the
 * unused parts is not initialized.
 */
class BitBoardSlice {
 public:
  BitBoardSlice(const uint32_t* rows, int num_rows, int row_offset);

  uint32_t get_row(int k) const { return rows[k]; }
  int num_rows() const { return num_rows_; }
  bool empty() const { return num_rows_ == 0; }
  int start_row() const { return start_row_; }
  auto get_set_locations() const;

 protected:
  uint32_t rows[kBoardDimension];
  int num_rows_;
  int start_row_;
};

#pragma pack(push, 1)
class Piece {
 public:
  Piece(piece_index_t index) : index_(index) {}
  piece_index_t operator() const { return index_; }
  const char* name() const;
  auto get_corners(corner_constraint_t) const;

 private:
  piece_index_t index_;
};
#pragma pack(pop)
static_assert(sizeof(Piece) == 1);

#pragma pack(push, 1)
class PieceOrientation {
 public:
  PieceOrientation(piece_orientation_index_t index) : index_(index) {}
  piece_orientation_index_t operator() const { return index_; }
  const uint8_t* row_masks() const;
  const uint8_t* adjacent_row_masks() const;
  const uint8_t* diagonal_row_masks() const;
  int height() const;
  int width() const;

 private:
  piece_orientation_index_t index_;
};
#pragma pack(pop)
static_assert(sizeof(PieceOrientation) == 1);

#pragma pack(push, 1)
class PieceOrientationCorner {
 public:
  PieceOrientationCorner(piece_orientation_corner_index_t index) : index_(index) {}
  piece_orientation_corner_index_t operator() const { return index_; }
  Piece to_piece() const;
  PieceOrientation to_piece_orientation() const;
  Location corner_offset() const;

  BitBoardSlice to_bitboard_mask(Location loc) const;  // returns empty slice if out-of-bounds
  BitBoardSlice to_adjacent_bitboard_mask(Location loc) const;  // assumes in-bounds
  BitBoardSlice to_diagonal_bitboard_mask(Location loc) const;  // assumes in-bounds

 private:
  piece_orientation_corner_index_t index_;
};
#pragma pack(pop)
static_assert(sizeof(PieceOrientationCorner) == 2);

/*
 * PieceMask is suitable for storing a subset of the 21 pieces of the game.
 */
#pragma pack(push, 1)
class PieceMask {
 public:
  auto get_unset_bits() const;
  void set(Piece piece) { mask_ |= 1 << piece; }
  void clear() { mask_ = 0; }

 private:
  uint32_t mask_;
};
#pragma pack(pop)
static_assert(sizeof(PieceMask) == 4);

namespace tables {

extern const _PieceData kPieceData[];
extern const _PieceOrientationData kPieceOrientationData[];
extern const _PieceOrientationCornerData kPieceOrientationCornerData[];
extern const uint8_t kPieceOrientationRowMasks[];
extern const piece_orientation_corner_index_t kPieceOrientationCornerArray[];

}  // tables

}  // namespace blokus

#include <inline/games/blokus/Types.inl>
