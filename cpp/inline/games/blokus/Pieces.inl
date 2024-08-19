#include <games/blokus/Pieces.hpp>

#include <util/Asserts.hpp>

#include <cstring>

namespace blokus {

inline void CornerConstraint::set(direction_t dir, bool value) {
  if (value) {
    mask_ |= 1 << dir;
  } else {
    mask_ &= ~(1 << dir);
  }
}

inline int CornerConstraint::get_count() const { return std::popcount(mask_); }

inline void Location::set(int8_t row, int8_t col) {
  this->row = row;
  this->col = col;
}

inline bool Location::valid() const {
  return row >= 0 && col >= 0;
}

inline piece_index_t PieceOrientationCorner::piece_index() const {
  return kPieceOrientationCornerIndexToPieceIndex[index_];
}

inline const char* Piece::name() const {
}

inline auto Piece::orientations() const {
}

inline auto Piece::get_corners(CornerConstraint constraint) const {
}

}  // namespace blokus
