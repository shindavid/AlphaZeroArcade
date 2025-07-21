#include <games/blokus/Types.hpp>

#include <util/AnsiCodes.hpp>

#include <format>

namespace blokus {

namespace detail {

struct PieceMaskRange {
  struct Iterator {
   public:
    explicit Iterator(uint32_t mask) : mask_(mask) {}

    bool operator==(Iterator other) const { return mask_ == other.mask_; }
    bool operator!=(Iterator other) const { return mask_ != other.mask_; }
    int operator*() const { return std::countr_zero(mask_); }

    Iterator& operator++() {
      mask_ &= mask_ - 1;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

   private:
    uint32_t mask_;
  };

  PieceMaskRange(uint32_t mask) : mask_(mask) {}

  Iterator begin() const { return Iterator(mask_); }
  Iterator end() const { return Iterator(0); }

 private:
  uint32_t mask_;
};

struct BitBoardRange {
  struct Iterator {
   public:
    Iterator(const BitBoard* bitboard, int row, int col)
        : bitboard_(bitboard), row_(row), col_(col) {
      skip_to_next();
    }

    bool operator==(Iterator other) const { return row_ == other.row_ && col_ == other.col_; }

    bool operator!=(Iterator other) const { return row_ != other.row_ || col_ != other.col_; }

    Location operator*() const { return Location{(int8_t)row_, (int8_t)col_}; }

    Iterator& operator++() {
      col_++;
      bool col_overflow = col_ == kBoardDimension;
      col_ *= !col_overflow;
      row_ += col_overflow;

      skip_to_next();
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

   private:
    void skip_to_next() {
      while (row_ < kBoardDimension && (bitboard_->get_row(row_) >> col_) == 0) {
        row_++;
        col_ = 0;
      }
      if (row_ < kBoardDimension) {
        col_ += std::countr_zero(bitboard_->get_row(row_) >> col_);
      }
    }

   private:
    const BitBoard* bitboard_;
    int row_;
    int col_;
  };

  BitBoardRange(const BitBoard* bitboard) : bitboard_(bitboard) {}

  Iterator begin() const { return Iterator(bitboard_, 0, 0); }
  Iterator end() const { return Iterator(bitboard_, kBoardDimension, 0); }

 private:
  const BitBoard* bitboard_;
};

struct BitBoardSliceRange {
  struct Iterator {
    Iterator(const BitBoardSlice* bitboard, int row, int col)
        : bitboard_(bitboard), row_(row), col_(col) {
      skip_to_next();
    }

    bool operator==(Iterator other) const { return row_ == other.row_ && col_ == other.col_; }

    bool operator!=(Iterator other) const { return row_ != other.row_ || col_ != other.col_; }

    Location operator*() const {
      return Location(row_, col_);
    }

    Iterator& operator++() {
      col_++;
      bool col_overflow = col_ == kBoardDimension;
      col_ *= !col_overflow;
      row_ += col_overflow;

      skip_to_next();
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

   private:
    void skip_to_next() {
      while (row_ < bitboard_->end_row() && (bitboard_->get_row(row_) >> col_) == 0) {
        row_++;
        col_ = 0;
      }
      if (row_ < bitboard_->end_row()) {
        col_ += std::countr_zero(bitboard_->get_row(row_) >> col_);
      }
    }

    const BitBoardSlice* bitboard_;
    int row_;
    int col_;
  };

  BitBoardSliceRange(const BitBoardSlice* bitboard) : bitboard_(bitboard) {}

  Iterator begin() const { return Iterator(bitboard_, bitboard_->start_row(), 0); }
  Iterator end() const { return Iterator(bitboard_, bitboard_->end_row(), 0); }

 private:
  const BitBoardSlice* bitboard_;
};

struct PieceOrientationCornerRange {
  struct Iterator {
    Iterator(int index) : index_(index) {}

    bool operator==(Iterator other) const { return index_ == other.index_; }
    bool operator!=(Iterator other) const { return index_ != other.index_; }
    PieceOrientationCorner operator*() const { return tables::kCornerConstraintArray[index_]; }

    Iterator& operator++() {
      index_++;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

   private:
    int index_;
  };

  PieceOrientationCornerRange(int start, int end) : start_(start), end_(end) {}

  Iterator begin() const { return Iterator(start_); }
  Iterator end() const { return Iterator(end_); }

 private:
  int start_;
  int end_;
};

}  // namespace detail

inline color_t char_to_color(char c) {
  switch (c) {
    case 'R':
      return kRed;
    case 'G':
      return kGreen;
    case 'B':
      return kBlue;
    case 'Y':
      return kYellow;
    default:
      return kNumColors;
  }
}

inline char color_to_char(color_t c) {
  switch (c) {
    case kRed:
      return 'R';
    case kGreen:
      return 'G';
    case kBlue:
      return 'B';
    case kYellow:
      return 'Y';
    default:
      return '?';
  }
}

inline void Location::set(int8_t r, int8_t c) {
  this->row = r;
  this->col = c;
}

inline bool Location::valid() const { return row >= 0 && col >= 0; }

inline std::string Location::to_string() const {
  return std::format("{}{}", 'A' + col, row + 1);
}

inline Location Location::from_string(const std::string& s) {
  Location loc(-1, -1);
  if (s.size() < 2) return loc;

  int r = std::atoi(s.c_str() + 1) - 1;
  if (r < 0 || r >= kBoardDimension) return loc;

  int c = int(s[0]) - 'A';
  if (c >= 0 && c < kBoardDimension) {
    loc.set(r, c);
    return loc;
  }

  c = int(s[0]) - 'a';
  if (c >= 0 && c < kBoardDimension) {
    loc.set(r, c);
    return loc;
  }

  return loc;
}

inline int Location::flatten() const {
  return row * kBoardDimension + col;
}

inline Location Location::unflatten(int k) {
  return {int8_t(k / kBoardDimension), int8_t(k % kBoardDimension)};
}

template<concepts::BitBoardLike Board>
inline BitBoard BitBoard::operator|(const Board& other) const {
  BitBoard result;
  int i;
  for (i = 0; i < other.start_row(); ++i) {
    result.rows_[i] = rows_[i];
  }
  for (i = other.start_row(); i < other.end_row(); ++i) {
    result.rows_[i] = rows_[i] | other.get_row(i);
  }
  for (i = other.end_row(); i < kBoardDimension; ++i) {
    result.rows_[i] = rows_[i];
  }
  return result;
}

template<concepts::BitBoardLike Board>
inline BitBoard BitBoard::operator&(const Board& other) const {
  BitBoard result;
  int i;
  for (i = 0; i < other.start_row(); ++i) {
    result.rows_[i] = 0;
  }
  for (i = other.start_row(); i < other.end_row(); ++i) {
    result.rows_[i] = rows_[i] & other.get_row(i);
  }
  for (i = other.end_row(); i < kBoardDimension; ++i) {
    result.rows_[i] = 0;
  }
  return result;
}

inline BitBoard BitBoard::operator~() const {
  BitBoard result;
  for (int i = 0; i < kBoardDimension; ++i) {
    result.rows_[i] = (~rows_[i]) & ((1 << kBoardDimension) - 1);
  }
  return result;
}

template <concepts::BitBoardLike Board>
inline BitBoard& BitBoard::operator|=(const Board& other) {
  for (int i = other.start_row(); i < other.end_row(); ++i) {
    rows_[i] |= other.get_row(i);
  }
  return *this;
}

template <concepts::BitBoardLike Board>
inline BitBoard& BitBoard::operator&=(const Board& other) {
  int i;
  for (i = 0; i < other.start_row(); ++i) {
    rows_[i] = 0;
  }
  for (i = other.start_row(); i < other.end_row(); ++i) {
    rows_[i] &= other.get_row(i);
  }
  for (i = other.end_row(); i < kBoardDimension; ++i) {
    rows_[i] = 0;
  }
  return *this;
}

inline bool BitBoard::any() const {
  for (int i = 0; i < kBoardDimension; ++i) {
    if (rows_[i]) return true;
  }
  return false;
}

inline void BitBoard::clear() { std::memset(rows_, 0, sizeof(rows_)); }

inline int BitBoard::count() const {
  int count = 0;
  for (int i = 0; i < kBoardDimension; ++i) {
    count += std::popcount(rows_[i]);
  }
  return count;
}

inline void BitBoard::clear_at_and_after(const Location& loc) {
  int r = loc.row;
  int c = loc.col;

  rows_[r] &= (1 << c) - 1;
  for (int i = r + 1; i < kBoardDimension; ++i) {
    rows_[i] = 0;
  }
}

inline bool BitBoard::get(int row, int col) const { return rows_[row] & (1 << col); }

inline void BitBoard::set(int row, int col) { rows_[row] |= (1 << col); }

template <concepts::BitBoardLike Board>
inline void BitBoard::unset(const Board& other) {
  for (int i = other.start_row(); i < other.end_row(); ++i) {
    rows_[i] &= ~other.get_row(i);
  }
}

inline auto BitBoard::get_set_locations() const { return detail::BitBoardRange(this); }

inline void BitBoard::write_to(std::bitset<kNumCells>& bitset) const {
  // TODO: optimize this
  for (Location loc : get_set_locations()) {
    bitset[loc.row * kBoardDimension + loc.col] = true;
  }
}

inline corner_constraint_t BitBoard::get_corner_constraint(Location loc) const {
  int row = loc.row;
  int col = loc.col;

  bool N_unblocked = (row < kBoardDimension - 1) && !get(row + 1, col);
  bool E_unblocked = (col < kBoardDimension - 1) && !get(row, col + 1);
  bool S_unblocked = (row > 0) && !get(row - 1, col);
  bool W_unblocked = (col > 0) && !get(row, col - 1);

  int count = int(N_unblocked) + int(E_unblocked) + int(S_unblocked) + int(W_unblocked);
  DEBUG_ASSERT(count <= 2);

  if (N_unblocked) {
    DEBUG_ASSERT(!S_unblocked);
    if (W_unblocked) {
      return ccNW;
    }
    if (E_unblocked) {
      return ccNE;
    }
    return ccN;
  }
  if (S_unblocked) {
    DEBUG_ASSERT(!N_unblocked);
    if (W_unblocked) {
      return ccSW;
    }
    if (E_unblocked) {
      return ccSE;
    }
    return ccS;
  }
  if (W_unblocked) {
    DEBUG_ASSERT(!E_unblocked);
    return ccW;
  }
  if (E_unblocked) {
    return ccE;
  }

  return ccNone;
}

template <concepts::BitBoardLike Board>
inline bool BitBoard::intersects(const Board& other) const {
  for (int i = other.start_row(); i < other.end_row(); ++i) {
    if (rows_[i] & other.get_row(i)) return true;
  }
  return false;
}

inline BitBoard BitBoard::adjacent_squares() const {
  constexpr int B = kBoardDimension;

  BitBoard result;
  result.rows_[0] = smear_row(rows_[0]) | rows_[1];

  for (int i = 1; i < B - 1; ++i) {
    result.rows_[i] = smear_row(rows_[i]) | rows_[i - 1] | rows_[i + 1];
  }

  result.rows_[B - 1] = smear_row(rows_[B - 1]) | rows_[B - 2];
  return result;
}

inline BitBoard BitBoard::diagonal_squares() const {
  constexpr int B = kBoardDimension;

  BitBoard result;
  result.rows_[0] = smear_row(rows_[1]);

  for (int i = 1; i < B - 1; ++i) {
    result.rows_[i] = smear_row(rows_[i - 1]) | smear_row(rows_[i + 1]);
  }

  result.rows_[B - 1] = smear_row(rows_[B - 2]);
  return result;
}

inline uint32_t BitBoard::smear_row(uint32_t row) {
  return ((row >> 1) | (row << 1)) & ((1 << kBoardDimension) - 1);
}

inline BitBoardSlice::BitBoardSlice(const uint32_t* rows, int num_rows, int row_offset) {
  num_rows_ = num_rows;
  start_row_ = row_offset;

  for (int i = 0; i < num_rows; ++i) {
    rows_[i + row_offset] = rows[i];
  }
}

inline uint32_t BitBoardSlice::get_row(int k) const {
  DEBUG_ASSERT(k >= start_row_ && k < start_row_ + num_rows_);
  return rows_[k];
}

inline auto BitBoardSlice::get_set_locations() const { return detail::BitBoardSliceRange(this); }

inline void BoardString::set(Location loc, drawing_t color) {
  DEBUG_ASSERT(colors_[loc.row][loc.col] == dBlankSpace);
  colors_[loc.row][loc.col] = color;
}

template<concepts::BitBoardLike Board>
inline void BoardString::set(const Board& board, drawing_t color) {
  for (Location loc : board.get_set_locations()) {
    set(loc, color);
  }
}

// inline const char* Piece::name() const { return tables::kPieceData[index_].name; }

inline auto Piece::get_corners(corner_constraint_t constraint) const {
  const auto& data = tables::kPieceData[index_];

  int start = data.corner_range_start;
  for (int c = 0; c < int(constraint); ++c) {
    start += data.subrange_lengths[(c + 3) / 4];
  }
  int end = start + data.subrange_lengths[(int(constraint) + 3) / 4];

  return detail::PieceOrientationCornerRange(start, end);
}

inline piece_orientation_index_t Piece::canonical_orientation() const {
  return tables::kPieceData[index_].canonical;
}

inline const uint8_t* PieceOrientation::row_masks() const {
  const auto& data = tables::kPieceOrientationData[index_];
  return &tables::kPieceOrientationRowMasks[data.mask_array_start_index];
}

inline const uint8_t* PieceOrientation::adjacent_row_masks() const {
  const auto& data = tables::kPieceOrientationData[index_];
  return &tables::kPieceOrientationRowMasks[data.mask_array_start_index + height()];
}

inline const uint8_t* PieceOrientation::diagonal_row_masks() const {
  const auto& data = tables::kPieceOrientationData[index_];
  return &tables::kPieceOrientationRowMasks[data.mask_array_start_index + 2 * height() + 2];
}

inline int PieceOrientation::height() const { return tables::kPieceOrientationData[index_].height; }

inline int PieceOrientation::width() const { return tables::kPieceOrientationData[index_].width; }

inline piece_orientation_corner_index_t PieceOrientation::canonical_corner() const {
  return tables::kPieceOrientationData[index_].canonical_poc;
}

inline Piece PieceOrientationCorner::to_piece() const {
  return tables::kPieceOrientationCornerData[index_].piece;
}

inline PieceOrientation PieceOrientationCorner::to_piece_orientation() const {
  return tables::kPieceOrientationCornerData[index_].piece_orientation;
}

inline Location PieceOrientationCorner::corner_offset() const {
  return tables::kPieceOrientationCornerData[index_].corner_offset;
}

inline std::string PieceOrientationCorner::name() const {
  // TODO: better name
  return std::to_string(index_);
}

inline BitBoardSlice PieceOrientationCorner::to_bitboard_mask(Location loc) const {
  PieceOrientation po = to_piece_orientation();
  const uint8_t* base_rows = po.row_masks();
  int height = po.height();
  int width = po.width();
  Location offset = corner_offset();

  // out-of-bounds checks
  int top_margin = kBoardDimension - loc.row + offset.row - height - 1;
  int bot_margin = loc.row - offset.row + 1;
  int left_margin = loc.col - offset.col + 1;
  int right_margin = kBoardDimension - loc.col + offset.col - width - 1;

  if (top_margin < 0 || bot_margin < 0 || left_margin < 0 || right_margin < 0) {
    return BitBoardSlice(nullptr, 0, 0);
  }

  uint32_t rows[height];
  if (loc.col < offset.col) {
    for (int i = 0; i < height; ++i) {
      rows[i] = uint32_t(base_rows[i]) >> (offset.col - loc.col);
    }
  } else {
    for (int i = 0; i < height; ++i) {
      rows[i] = uint32_t(base_rows[i]) << (loc.col - offset.col);
    }
  }

  return BitBoardSlice(rows, height, bot_margin);
}

inline BitBoardSlice PieceOrientationCorner::to_adjacent_bitboard_mask(Location loc) const {
  PieceOrientation po = to_piece_orientation();
  int height = po.height();
  int width = po.width();
  Location offset = corner_offset();

  int top_margin = kBoardDimension - loc.row + offset.row - height - 1;
  int bot_margin = loc.row - offset.row + 1;
  int left_margin = loc.col - offset.col + 1;
  int right_margin = kBoardDimension - loc.col + offset.col - width - 1;

  DEBUG_ASSERT(top_margin >= 0);
  DEBUG_ASSERT(bot_margin >= 0);
  DEBUG_ASSERT(left_margin >= 0);
  DEBUG_ASSERT(right_margin >= 0);

  bool top_overflow = top_margin == 0;
  bool bot_overflow = bot_margin == 0;

  DEBUG_ASSERT(!(top_overflow && bot_overflow));

  const uint8_t* base_rows = po.adjacent_row_masks();
  int n_rows = height + !top_overflow +!bot_overflow;

  uint32_t rows[n_rows];
  if (left_margin == 0) {
    for (int i = 0; i < n_rows; ++i) {
      uint32_t base = uint32_t(base_rows[i + bot_overflow]);
      rows[i] = base >> 1;
    }
  } else {
    for (int i = 0; i < n_rows; ++i) {
      uint32_t base = uint32_t(base_rows[i + bot_overflow]);
      rows[i] = (base << (left_margin - 1)) & ((1 << kBoardDimension) - 1);
    }
  }

  return BitBoardSlice(rows, n_rows, bot_margin - !bot_overflow);
}

inline BitBoardSlice PieceOrientationCorner::to_diagonal_bitboard_mask(Location loc) const {
  PieceOrientation po = to_piece_orientation();
  int height = po.height();
  int width = po.width();
  Location offset = corner_offset();

  int top_margin = kBoardDimension - loc.row + offset.row - height - 1;
  int bot_margin = loc.row - offset.row + 1;
  int left_margin = loc.col - offset.col + 1;
  int right_margin = kBoardDimension - loc.col + offset.col - width - 1;

  DEBUG_ASSERT(top_margin >= 0);
  DEBUG_ASSERT(bot_margin >= 0);
  DEBUG_ASSERT(left_margin >= 0);
  DEBUG_ASSERT(right_margin >= 0);

  bool top_overflow = top_margin == 0;
  bool bot_overflow = bot_margin == 0;

  DEBUG_ASSERT(!(top_overflow && bot_overflow));

  const uint8_t* base_rows = po.diagonal_row_masks();
  int n_rows = height + !top_overflow + !bot_overflow;

  uint32_t rows[n_rows];
  if (left_margin == 0) {
    for (int i = 0; i < n_rows; ++i) {
      uint32_t base = uint32_t(base_rows[i + bot_overflow]);
      rows[i] = base >> 1;
    }
  } else {
    for (int i = 0; i < n_rows; ++i) {
      uint32_t base = uint32_t(base_rows[i + bot_overflow]);
      rows[i] = (base << (left_margin - 1)) & ((1 << kBoardDimension) - 1);
    }
  }

  return BitBoardSlice(rows, n_rows, bot_margin - !bot_overflow);
}

inline Location PieceOrientationCorner::get_root_location(Location loc) const {
  BitBoardSlice slice = to_bitboard_mask(loc);
  for (Location root_loc : slice.get_set_locations()) {
    return root_loc;
  }
  throw util::Exception("Unexpected empty slice");
}

inline PieceMask PieceMask::operator~() const {
  PieceMask result;
  result.mask_ = (~mask_) & ((1 << kNumPieces) - 1);
  return result;
}

inline PieceMask PieceMask::operator&(const PieceMask& other) const {
  PieceMask result;
  result.mask_ = mask_ & other.mask_;
  return result;
}

inline PieceMask& PieceMask::operator&=(const PieceMask& other) {
  mask_ &= other.mask_;
  return *this;
}

inline auto PieceMask::get_set_bits() const {
  return detail::PieceMaskRange(mask_);
}

inline auto PieceMask::get_unset_bits() const {
  uint32_t unset_bits = ~mask_ & ((1 << kNumPieces) - 1);
  return detail::PieceMaskRange(unset_bits);
}

}  // namespace blokus
