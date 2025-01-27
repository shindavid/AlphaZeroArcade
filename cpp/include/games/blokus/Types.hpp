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

// Converts {'B', 'Y', 'R', 'G', *} to {kBlue, kYellow, kRed, kGreen, kNumColors}
color_t char_to_color(char c);

// Converts {kBlue, kYellow, kRed, kGreen, kNumColors} to {'B', 'Y', 'R', 'G', '?'}
char color_to_char(color_t c);

struct Location {
  Location() = default;

  template<typename R, typename C> Location(R r, C c) : row(r), col(c) {}

  auto operator<=>(const Location& other) const = default;
  void set(int8_t r, int8_t c);
  bool valid() const;
  std::string to_string() const;
  static Location from_string(const std::string& s);  // return invalid Location on failure

  int flatten() const;
  static Location unflatten(int k);

  int8_t row;
  int8_t col;
};
static_assert(sizeof(Location) == 2);

struct _MiniBoardLookup {
  uint32_t key;
  piece_orientation_corner_index_t value;
};
static_assert(sizeof(_MiniBoardLookup) == 8);

// Backs the Piece type
struct _PieceData {
  piece_orientation_index_t canonical;
  int8_t subrange_lengths[3];  // indexed by number of unblocked directions

  /*
   * Indexes into tables::kCornerConstraintArray. Starting from this point, the array contains
   * subsequences for corner_constraint_t(0), corner_constraint_t(1), ...
   *
   * The length of each subsequence can be found in num_oriented_corners.
   */
  int16_t corner_range_start;
};
static_assert(sizeof(_PieceData) == 6);

// Backs the PieceOrientation type
struct _PieceOrientationData {
  int16_t mask_array_start_index;
  piece_orientation_corner_index_t canonical_poc;
  int8_t width;
  int8_t height;
};
static_assert(sizeof(_PieceOrientationData) == 6);

// Backs the PieceOrientationCorner type
struct _PieceOrientationCornerData {
  Location corner_offset;
  piece_index_t piece;
  piece_orientation_index_t piece_orientation;
};
static_assert(sizeof(_PieceOrientationCornerData) == 4);

namespace concepts {

template <class Board>
concept BitBoardLike = requires(const Board& board, int k) {
  { board.start_row() } -> std::same_as<int>;
  { board.end_row() } -> std::same_as<int>;
  { board.get_row(k) } -> std::same_as<uint32_t>;
};

}  // namespace concepts

class BitBoardSlice;

/*
 * BitBoard is suitable for storing a board-mask for the entire board
 */
class BitBoard {
 public:
  auto operator<=>(const BitBoard& other) const = default;

  template<concepts::BitBoardLike Board>
  BitBoard operator|(const Board& other) const;

  template<concepts::BitBoardLike Board>
  BitBoard operator&(const Board& other) const;

  BitBoard operator~() const;

  template <concepts::BitBoardLike Board>
  BitBoard& operator&=(const Board& other);

  template<concepts::BitBoardLike Board>
  BitBoard& operator|=(const Board& other);

  constexpr int start_row() const { return 0; }
  constexpr int end_row() const { return kBoardDimension; }

  std::string to_string(drawing_t) const;
  bool any() const;
  void clear();
  int count() const;
  void clear_at_and_after(const Location& loc);
  uint32_t get_row(int k) const { return rows_[k]; }
  bool get(int row, int col) const;
  bool get(const Location& loc) const { return get(loc.row, loc.col); }
  void set(int row, int col);
  void set(const Location& loc) { set(loc.row, loc.col); }

  template <concepts::BitBoardLike Board>
  void unset(const Board&);

  auto get_set_locations() const;
  void write_to(std::bitset<kNumCells>& bitset) const;
  corner_constraint_t get_corner_constraint(Location loc) const;

  template <concepts::BitBoardLike Board>
  bool intersects(const Board& other) const;

  /*
   * Requires that this is set at loc, and that none of the lexicographically smaller set squares
   * are connected to loc.
   *
   * Crawls to find the connected subset of squares containing loc, matches that to a specific
   * PieceOrientation po, and returns the corresponding piece_orientation_corner_index_t.
   */
  piece_orientation_corner_index_t find(Location loc) const;

  BitBoard adjacent_squares() const;  // All adjacent-neighbors of set squares
  BitBoard diagonal_squares() const;  // All diagonal-neighbors of set squares

 protected:
  static uint32_t smear_row(uint32_t);  // Returns 1-dimensional L1-neighbors of distance 1
  uint32_t rows_[kBoardDimension];
};
static_assert(concepts::BitBoardLike<BitBoard>);

/*
 * BitBoardSlice uses only a subset of the rows of the board. The memory corresponding to the
 * unused parts is not initialized.
 */
class BitBoardSlice {
 public:
  BitBoardSlice(const uint32_t* rows, int num_rows, int row_offset);

  uint32_t get_row(int k) const;
  bool empty() const { return num_rows_ == 0; }
  int start_row() const { return start_row_; }
  int end_row() const { return start_row_ + num_rows_; }
  auto get_set_locations() const;

 protected:
  uint32_t rows_[kBoardDimension];
  int num_rows_;
  int start_row_;
};
static_assert(concepts::BitBoardLike<BitBoardSlice>);

class BoardString {
 public:
  void print(std::ostream&, bool omit_trivial_rows = false) const;
  void pretty_print(std::ostream&) const;

  void set(Location loc, drawing_t color);

  template<concepts::BitBoardLike Board>
  void set(const Board& board, drawing_t color);

 private:
  drawing_t colors_[kBoardDimension][kBoardDimension] = {};
};

class Piece;
class PieceOrientation;

class TuiPrompt {
 public:
  friend class Piece;
  friend class PieceOrientation;

  void print();

 private:
  static constexpr int kNumLines = 7;

  struct Block {
    int width;
    std::ostringstream lines[kNumLines];
  };
  using block_vec_t = std::vector<Block>;

  block_vec_t blocks_;
};

class Piece {
 public:
  Piece(piece_index_t index) : index_(index) {}
  operator piece_index_t() const { return index_; }
  // const char* name() const;
  auto get_corners(corner_constraint_t) const;
  piece_orientation_index_t canonical_orientation() const;
  void write_to(TuiPrompt& prompt, color_t) const;

 private:
  piece_index_t index_;
};
static_assert(sizeof(Piece) == 1);

class PieceOrientation {
 public:
  PieceOrientation(piece_orientation_index_t index) : index_(index) {}
  operator piece_orientation_index_t() const { return index_; }
  const uint8_t* row_masks() const;
  const uint8_t* adjacent_row_masks() const;
  const uint8_t* diagonal_row_masks() const;
  int height() const;
  int width() const;
  piece_orientation_corner_index_t canonical_corner() const;
  void write_to(TuiPrompt& prompt, color_t, int label) const;

 private:
  piece_orientation_index_t index_;
};
static_assert(sizeof(PieceOrientation) == 1);

class PieceOrientationCorner {
 public:
  PieceOrientationCorner(piece_orientation_corner_index_t index) : index_(index) {}
  operator piece_orientation_corner_index_t() const { return index_; }
  Piece to_piece() const;
  PieceOrientation to_piece_orientation() const;
  Location corner_offset() const;
  std::string name() const;

  void pretty_print(std::ostream&, color_t) const;

  static PieceOrientationCorner from_action(core::action_t a) { return a; }
  core::action_t to_action() const { return index_; }

  BitBoardSlice to_bitboard_mask(Location loc) const;  // returns empty slice if out-of-bounds
  BitBoardSlice to_adjacent_bitboard_mask(Location loc) const;  // assumes in-bounds
  BitBoardSlice to_diagonal_bitboard_mask(Location loc) const;  // assumes in-bounds

  /*
   * If this is positioned at loc, returns the root location of the piece, which is defined as the
   * lexically smallest location of the piece.
   */
  Location get_root_location(Location loc) const;

 private:
  piece_orientation_corner_index_t index_;
};
static_assert(sizeof(PieceOrientationCorner) == 2);

/*
 * PieceMask is suitable for storing a subset of the 21 pieces of the game.
 */
class PieceMask {
 public:
  auto operator<=>(const PieceMask& other) const = default;

  PieceMask operator~() const;
  PieceMask operator&(const PieceMask&) const;
  PieceMask& operator&=(const PieceMask&);
  auto get_set_bits() const;
  auto get_unset_bits() const;
  int count() const { return std::popcount(mask_); }
  bool empty() const { return mask_ == 0; }
  bool get(Piece piece) const { return mask_ & (1 << piece); }
  void set(Piece piece) { mask_ |= 1 << piece; }
  void clear() { mask_ = 0; }

 private:
  uint32_t mask_;
};
static_assert(sizeof(PieceMask) == 4);

namespace tables {

/*
 * The following tables are computed by py/games/blokus/cpp_writer.py
 *
 * Memory size:
 *
 * kPieceData: 8 * 21 = 168 bytes
 * kPieceOrientationData: 4 * 91 = 364 bytes
 * kPieceOrientationCornerData: 4 * 309 = 1236 bytes
 * kPieceOrientationRowMasks: 1 * 1102 = 1102 bytes
 * kCornerConstraintArray: 2 * 709 = 1418 bytes
 *
 * Total: 4288 bytes
 */
extern const _PieceData kPieceData[kNumPieces];
extern const _PieceOrientationData kPieceOrientationData[kNumPieceOrientations];
extern const _PieceOrientationCornerData kPieceOrientationCornerData[kNumPieceOrientationCorners];
extern const _MiniBoardLookup kMiniBoardLookup[kMiniBoardLookupSize];
extern const uint8_t kPieceOrientationRowMasks[kNumPieceOrientationRowMasks];
extern const piece_orientation_corner_index_t kCornerConstraintArray[kCornerConstraintArraySize];

}  // tables

}  // namespace blokus

#include <inline/games/blokus/Types.inl>
