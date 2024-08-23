#include <games/blokus/Types.hpp>

#include <iomanip>

namespace blokus {

namespace detail {

struct PieceMaskRange {
  struct Iterator {
   public:
    explicit Iterator(uint32_t mask) : mask(mask) {}

    bool operator==(Iterator other) const { return mask == other.mask; }
    bool operator!=(Iterator other) const { return mask != other.mask; }
    int operator*() const { return std::countr_zero(mask); }

    Iterator& operator++() {
      mask &= mask - 1;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

   private:
    uint32_t mask;
  };

  PieceMaskRange(uint32_t mask) : mask(mask) {}

  Iterator begin() const { return Iterator(mask); }
  Iterator end() const { return Iterator(0); }

 private:
  uint32_t mask;
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
      return Location{int8_t(row_), (int8_t)col_};
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

inline void Location::set(int8_t row, int8_t col) {
  this->row = row;
  this->col = col;
}

inline bool Location::valid() const { return row >= 0 && col >= 0; }

inline int Location::flatten() const {
  return row * kBoardDimension + col;
}

inline Location Location::unflatten(int k) {
  return {int8_t(k / kBoardDimension), int8_t(k % kBoardDimension)};
}

inline BitBoard BitBoard::operator|(const BitBoard& other) const {
  BitBoard result;
  for (int i = 0; i < kBoardDimension; ++i) {
    result.rows_[i] = rows_[i] | other.rows_[i];
  }
  return result;
}

inline BitBoard& BitBoard::operator&=(const BitBoard& other) {
  for (int i = 0; i < kBoardDimension; ++i) {
    rows_[i] &= other.rows_[i];
  }
  return *this;
}

inline BitBoard BitBoard::operator~() const {
  BitBoard result;
  for (int i = 0; i < kBoardDimension; ++i) {
    result.rows_[i] = (~rows_[i]) & ((1 << kBoardDimension) - 1);
  }
  return result;
}

inline BitBoard& BitBoard::operator|=(const BitBoard& other) {
  for (int i = 0; i < kBoardDimension; ++i) {
    rows_[i] |= other.get_row(i);
  }
  return *this;
}

inline BitBoard& BitBoard::operator|=(const BitBoardSlice& other) {
  for (int i = other.start_row(); i < other.end_row(); ++i) {
    rows_[i] |= other.get_row(i);
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

inline void BitBoard::set(const Location& loc) { set(loc.row, loc.col); }

inline void BitBoard::unset(const BitBoard& other) {
  for (int i = 0; i < kBoardDimension; ++i) {
    rows_[i] &= ~other.rows_[i];
  }
}

inline void BitBoard::unset(const BitBoardSlice& other) {
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
  util::debug_assert(count <= 2);

  if (N_unblocked) {
    util::debug_assert(!S_unblocked);
    if (W_unblocked) {
      return ccNW;
    }
    if (E_unblocked) {
      return ccNE;
    }
    return ccN;
  }
  if (S_unblocked) {
    util::debug_assert(!N_unblocked);
    if (W_unblocked) {
      return ccSW;
    }
    if (E_unblocked) {
      return ccSE;
    }
    return ccS;
  }
  if (W_unblocked) {
    util::debug_assert(!E_unblocked);
    return ccW;
  }
  if (E_unblocked) {
    return ccE;
  }

  return ccNone;
}

inline bool BitBoard::intersects(const BitBoardSlice& other) const {
  for (int i = other.start_row(); i < other.end_row(); ++i) {
    if (rows_[i] & other.get_row(i)) return true;
  }
  return false;
}

inline piece_orientation_corner_index_t BitBoard::find(Location loc) const {
  throw std::runtime_error("Not implemented");
}

inline BitBoard BitBoard::operator&(const BitBoard& other) const {
  throw std::runtime_error("Not implemented");
}

inline BitBoard BitBoard::adjacent_squares() const {
  constexpr int B = kBoardDimension;

  BitBoard result;
  result.rows_[0] = (rows_[0] >> 1);
  result.rows_[0] |= (result.rows_[0] << 2);
  result.rows_[0] |= rows_[1];

  for (int i = 1; i < B - 1; ++i) {
    result.rows_[i] = rows_[i] >> 1;
    result.rows_[i] |= (result.rows_[i] << 2);
    result.rows_[i] |= rows_[i - 1];
    result.rows_[i] |= rows_[i + 1];
  }

  result.rows_[B - 1] = (rows_[B - 1] >> 1);
  result.rows_[B - 1] |= (result.rows_[B - 1] << 2);
  result.rows_[B - 1] |= rows_[B - 2];
  return result;
}

inline BitBoard BitBoard::diagonal_squares() const {
  constexpr int B = kBoardDimension;

  BitBoard result;
  result.rows_[0] = (rows_[1] >> 1);
  result.rows_[0] |= (result.rows_[0] << 2);

  for (int i = 1; i < B - 1; ++i) {
    uint32_t row1 = rows_[i - 1];
    uint32_t row2 = rows_[i + 1];

    uint32_t row3 = row1 >> 1;
    row3 |= (row3 << 2);

    uint32_t row4 = row2 >> 1;
    row4 |= (row4 << 2);

    result.rows_[i] = row3 | row4;
  }

  result.rows_[B - 1] = (rows_[B - 2] >> 1);
  result.rows_[B - 1] |= (result.rows_[B - 1] << 2);
  return result;
}

inline BitBoardSlice::BitBoardSlice(const uint32_t* rows, int num_rows, int row_offset) {
  num_rows_ = num_rows;
  start_row_ = row_offset;

  for (int i = 0; i < num_rows; ++i) {
    rows_[i + row_offset] = rows[i];
  }
}

inline uint32_t BitBoardSlice::get_row(int k) const {
  util::debug_assert(k >= start_row_ && k < start_row_ + num_rows_);
  return rows_[k];
}

inline auto BitBoardSlice::get_set_locations() const { return detail::BitBoardSliceRange(this); }

inline BoardString::BoardString() {
  for (int i = 0; i < kBoardDimension; ++i) {
    for (int j = 0; j < kBoardDimension; ++j) {
      strs_[i][j] = std::string(".");
    }
  }
}

inline void BoardString::print(std::ostream& os, bool omit_trivial_rows) const {
  os << "   ";
  for (int col = 0; col < kBoardDimension; ++col) {
    os << static_cast<char>('A' + col);
  }
  os << '\n';
  for (int row = kBoardDimension - 1; row >= 0; --row) {
    bool trivial = true;
    for (int col = 0; col < kBoardDimension; ++col) {
      if (strs_[row][col] != ".") {
        trivial = false;
        break;
      }
    }
    if (omit_trivial_rows && trivial) continue;

    os << std::setw(2) << (row + 1) << ' ';
    for (int col = 0; col < kBoardDimension; ++col) {
      os << strs_[row][col];
    }
    os << ' ' << std::setw(2) << (row + 1) << '\n';
  }
  os << "   ";
  for (int col = 0; col < kBoardDimension; ++col) {
    os << static_cast<char>('A' + col);
  }
  os << '\n';
}

inline void BoardString::set(Location loc, const std::string& str) {
  util::debug_assert(strs_[loc.row][loc.col] == ".");
  strs_[loc.row][loc.col] = str;
}

inline void BoardString::set(const BitBoard& board, const std::string& str) {
  for (Location loc : board.get_set_locations()) {
    set(loc, str);
  }
}

inline void BoardString::set(const BitBoardSlice& board, const std::string& str) {
  for (Location loc : board.get_set_locations()) {
    set(loc, str);
  }
}

inline const char* Piece::name() const { return tables::kPieceData[index_].name; }

inline auto Piece::get_corners(corner_constraint_t constraint) const {
  const auto& data = tables::kPieceData[index_];

  int start = data.corner_range_start;
  for (int c = 0; c < int(constraint); ++c) {
    start += data.subrange_lengths[(c + 3) / 4];
  }
  int end = start + data.subrange_lengths[(int(constraint) + 3) / 4];

  return detail::PieceOrientationCornerRange(start, end);
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

inline PieceOrientationCorner PieceOrientationCorner::from_action(core::action_t a) {
  util::debug_assert(a >= kNumCells + 1 && a < kNumActions);
  return a - kNumCells - 1;
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

  util::debug_assert(top_margin >= 0);
  util::debug_assert(bot_margin >= 0);
  util::debug_assert(left_margin >= 0);
  util::debug_assert(right_margin >= 0);

  bool top_overflow = top_margin == 0;
  bool bot_overflow = bot_margin == 0;

  util::debug_assert(!(top_overflow && bot_overflow));

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

  util::debug_assert(top_margin >= 0);
  util::debug_assert(bot_margin >= 0);
  util::debug_assert(left_margin >= 0);
  util::debug_assert(right_margin >= 0);

  bool top_overflow = top_margin == 0;
  bool bot_overflow = bot_margin == 0;

  util::debug_assert(!(top_overflow && bot_overflow));

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

inline auto PieceMask::get_unset_bits() const {
  uint32_t unset_bits = ~mask_ & ((1 << kNumPieces) - 1);
  return detail::PieceMaskRange(unset_bits);
}

}  // namespace blokus
