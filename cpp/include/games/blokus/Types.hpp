#pragma once

#include <core/BasicTypes.hpp>
#include <games/blokus/Constants.hpp>
#include <util/FiniteGroups.hpp>

#include <bit>
#include <bitset>
#include <cstdint>

namespace blokus {

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
  CornerConstraint get_corner_constraint(Location loc) const;
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
  int start_row() const { return start_row_; }
  auto get_set_locations() const;

 protected:
  uint32_t rows[kBoardDimension];
  int num_rows_;
  int start_row_;
};

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

}  // namespace blokus

#include <inline/games/blokus/Types.inl>
