#include <games/blokus/BitBoard.hpp>

#include <cstring>

namespace blokus {

namespace detail {

struct PieceMaskWrapper {
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

  PieceMaskWrapper(uint32_t mask) : mask(mask) {}

  Iterator begin() const { return Iterator(mask); }
  Iterator end() const { return Iterator(0); }

  uint32_t mask;
};

struct BitBoardWrapper {
  struct Iterator {
   public:
    Iterator(const BitBoard* bitboard, int row, int col)
        : bitboard_(bitboard), row_(row), col_(col) {
      skip_to_next();
    }

    bool operator==(Iterator other) const {
      return row_ == other.row_ && col_ == other.col_;
    }

    bool operator!=(Iterator other) const {
      return row_ != other.row_ || col_ != other.col_;
    }

    Location operator*() const { return Location{row_, col_}; }

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
      while (row_ < kBoardDimension && !bitboard_->get_row(row_)) {
        row_++;
        col_ = 0;
      }
      if (row_ < kBoardDimension) {
        col_ = std::countr_zero(bitboard_->get_row(row_));
      }
    }

    const BitBoard* bitboard_;
    int row_;
    int col_;
  };

  BitBoardWrapper(const BitBoard* bitboard) : bitboard_(bitboard) {}

  Iterator begin() const { return Iterator(bitboard_, 0, 0); }
  Iterator end() const { return Iterator(bitboard_, kBoardDimension, 0); }

  const BitBoard* bitboard_;
};


struct BitBoardSliceWrapper {
  struct Iterator {
    Iterator(const BitBoardSlice* bitboard, int row, int col)
        : bitboard_(bitboard), row_(row), col_(col) {
      skip_to_next();
    }

    bool operator==(Iterator other) const {
      return row_ == other.row_ && col_ == other.col_;
    }

    bool operator!=(Iterator other) const {
      return row_ != other.row_ || col_ != other.col_;
    }

    Location operator*() const { return Location{row_ + bitboard_.num_rows(), col_}; }

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
      while (row_ < bitboard_->num_rows() && !bitboard_->get_row(row_)) {
        row_++;
        col_ = 0;
      }
      if (row_ < bitboard_->num_rows()) {
        col_ = std::countr_zero(bitboard_->get_row(row_));
      }
    }

    const BitBoardSlice* bitboard_;
    int row_;
    int col_;
  };

  BitBoardSliceWrapper(const BitBoardSlice* bitboard) : bitboard_(bitboard) {}

  Iterator begin() const { return Iterator(bitboard_, 0, 0); }
  Iterator end() const { return Iterator(bitboard_, bitboard_->num_rows(), 0); }

  const BitBoardSlice* bitboard_;
};

}  // namespace detail

inline auto PieceMask::get_unset_bits() const {
  uint32_t unset_bits = ~mask_ & ((1 << kNumPieces) - 1);
  return detail::PieceMaskWrapper(unset_bits);
}

inline BoardString::BoardString() {
  for (int i = 0; i < kBoardDimension; ++i) {
    for (int j = 0; j < kBoardDimension; ++j) {
      strs_[i * kBoardDimension + j] = ".";
    }
  }
}

inline void BoardString::print(std::ostream& os) const {
  os << "   ";
  for (int col = 0; col < kBoardDimension; ++col) {
    os << static_cast<char>('A' + col);
  }
  os << '\n';
  for (int row = kBoardDimension - 1; row >= 0; --row) {
    os << std::setw(2) << (row + 1) << ' ';
    for (int col = 0; col < kBoardDimension; ++col) {
      os << strs_[row * kBoardDimension + col];
    }
    os << ' ' << std::setw(2) << (row + 1) << '\n';
  }
  os << "   ";
  for (int col = 0; col < kBoardDimension; ++col) {
    os << static_cast<char>('A' + col);
  }
  os << '\n';
}

inline void BoardString::set(const BitBoard& board, const std::string& str) {
  for (Location loc : board.get_set_locations()) {
    strs_[loc.row * kBoardDimension + loc.col] = str;
  }
}

inline void BoardString::set(const BitBoardSlice& board, const std::string& str) {
  for (Location loc : board.get_set_locations()) {
    strs_[loc.row * kBoardDimension + loc.col] = str;
  }
}

inline BitBoard BitBoard::operator|(const BitBoard& other) const {
  BitBoard result;
  for (int i = 0; i < kBoardDimension; ++i) {
    result.rows[i] = rows[i] | other.rows[i];
  }
  return result;
}

inline BitBoard& BitBoard::operator&=(const BitBoard& other) {
  for (int i = 0; i < kBoardDimension; ++i) {
    rows[i] &= other.rows[i];
  }
  return *this;
}

inline BitBoard BitBoard::operator~() const {
  BitBoard result;
  for (int i = 0; i < kBoardDimension; ++i) {
    result.rows[i] = (~rows[i]) & ((1 << kBoardDimension) - 1);
  }
  return result;
}

inline BitBoard& BitBoard::operator|=(const BitBoardSlice& other) {
  for (int i = 0; i < other.num_rows(); ++i) {
    rows[i + other.start_row()] |= other.get_row(i);
  }
  return *this;
}

inline bool BitBoard::any() const {
  for (int i = 0; i < kBoardDimension; ++i) {
    if (rows[i]) return true;
  }
  return false;
}

inline void BitBoard::clear() { std::memset(rows, 0, sizeof(rows)); }

inline int BitBoard::count() {
  int count = 0;
  for (int i = 0; i < kBoardDimension; ++i) {
    count += std::popcount(rows[i]);
  }
  return count;
}

inline void BitBoard::clear_at_and_after(const Location& loc) {
  int r = loc.row;
  int c = loc.col;

  rows[r] &= (1 << c) - 1;
  for (int i = r + 1; i < kBoardDimension; ++i) {
    rows[i] = 0;
  }
}

inline bool BitBoard::get(int row, int col) const {
  return rows[row] & (1 << col);
}

inline void BitBoard::set(int row, int col) {
  rows[row] |= (1 << col);
}

inline void BitBoard::set(const Location& loc) {
  set(loc.row, loc.col);
}

inline auto BitBoard::get_set_locations() const {
  return detail::BitBoardWrapper(this);
}

inline void BitBoard::write_to(std::bitset<kNumCells>& bitset) const {
  // TODO: optimize this
  for (Location loc : get_set_locations()) {
    bitset[loc.row * kBoardDimension + loc.col] = true;
  }
}

inline CornerConstraint BitBoard::get_corner_constraint(Location loc) const {
  int row = loc.row;
  int col = loc.col;

  CornerConstraint constraint;

  if (row < kBoardDimension - 1) constraint.set(dN, !get(row + 1, col));
  if (col < kBoardDimension - 1) constraint.set(dE, !get(row, col + 1));
  if (row > 0) constraint.set(dS, !get(row - 1, col));
  if (col > 0) constraint.set(dW, !get(row, col - 1));

  return constraint;
}

inline bool BitBoard::intersects(const BitBoardSlice& other) const {
  for (int i = 0; i < other.num_rows(); ++i) {
    if (rows[i + other.start_row()] & other.get_row(i)) return true;
  }
  return false;
}

inline BitBoardSlice::BitBoardSlice(const uint32_t* rows, int num_rows, int row_offset) {
  num_rows_ = num_rows;
  start_row_ = row_offset;

  for (int i = 0; i < num_rows; ++i) {
    rows_[i + row_offset] = rows[i];
  }
}

inline auto BitBoardSlice::get_set_locations() const { return detail::BitBoardSliceWrapper(this); }

// inline BitBoardSlice BitBoardSlice::adjacent_neighbors() const {
//   bool bottom_overflow = start_row_ == 0;
//   bool top_overflow = start_row_ + num_rows_ == kBoardDimension;
//   util::debug_assert(!(bottom_overflow && top_overflow));

//   int num_new_rows = num_rows_ + !bottom_overflow + !top_overflow;
//   uint32_t new_rows[num_new_rows] = {};

//   // down-shift
//   for (int i = bottom_overflow; i < num_rows_; ++i) {
//     new_rows[i - bottom_overflow] |= rows_[i];
//   }

//   // up-shift
//   for (int i = 0; i < num_rows_ - top_overflow; ++i) {
//     new_rows[i + 2 - bottom_overflow] |= rows_[i];
//   }

//   // left/right-shift
//   for (int i = 0; i < num_rows_; ++i) {
//     uint32_t row = rows_[i];
//     new_rows[i + !bottom_overflow] |= ((row >> 1) | (row << 1)) & ((1 << kBoardDimension) - 1);
//   }

//   // remove original rows
//   for (int i = 0; i < num_rows_; ++i) {
//     new_rows[i + !bottom_overflow] &= ~rows_[i];
//   }

//   return BitBoardSlice(new_rows, num_new_rows, start_row_ - !bottom_overflow);
// }

}  // namespace blokus
