#include <games/blokus/Types.hpp>

#include <util/Asserts.hpp>

#include <bit>

namespace blokus {

namespace detail {

// MiniBoard is a specialized struct that is just large enough to store the location of a single
// piece on the board.
//
// loc_ corresponds to the board square that is lexically smallest among the squares occupied by the
// piece.
//
// The remaining occupied squares are stored in mask_, whose indices are given by the following
// diagram:
//
//          28
//       20 21 22
//    12 13 14 15 16
//  4  5  6  7  8  9 10
//           L  0  1  2  3
class MiniBoard {
 public:
  MiniBoard(Location loc) : loc_(loc) {}

  bool any() const { return mask_; }

  Location pop() {
    if (!any()) {
      throw util::Exception("MiniBoard::pop() called on empty MiniBoard");
    }

    int i = std::countr_zero(mask_);
    mask_ &= ~(1 << i);
    return index_to_location(i);
  }

  void set(Location loc) {
    int i = location_to_index(loc);
    mask_ |= 1 << i;
  }

  bool get(Location loc) const {
    int i = location_to_index(loc);
    return mask_ & (1 << i);
  }

  bool in_bounds(Location loc) const {
    bool row_ok = loc.row >= row_lower_bound() && loc.row < row_upper_bound();
    bool col_ok = loc.col >= col_lower_bound() && loc.col < col_upper_bound();
    return row_ok && col_ok;
  }

  piece_orientation_corner_index_t to_piece_orientation_corner_index() const {
    // binary search over tables::kMiniBoardLookup:
    int left = 0;
    int right = kMiniBoardLookupSize;
    while (left < right) {
      int mid = (left + right) / 2;
      if (tables::kMiniBoardLookup[mid].key < mask_) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    if (left == kMiniBoardLookupSize || tables::kMiniBoardLookup[left].key != mask_) {
      throw util::Exception(
          "MiniBoard::to_piece_orientation_corner_index() failed to find key %08x", mask_);
    }
    return tables::kMiniBoardLookup[left].value;
  }

 private:
  Location index_to_location(int i) const {
    int r = (i + 4) / 8;
    int c = (i + r + 2 + 2 * (i < 4)) % 8 - 3;
    return {int8_t(loc_.row + r), int8_t(loc_.col + c)};
  }

  int location_to_index(Location loc) const {
    int r = loc.row - loc_.row;
    int c = loc.col - loc_.col;
    return 7 * r + c + 1 - 2*(r < 1);
  }

  int8_t row_lower_bound() const { return loc_.row; }
  int8_t row_upper_bound() const { return int8_t(std::min(loc_.row + 5, kBoardDimension)); }
  int8_t col_lower_bound() const { return int8_t(std::max(0, loc_.col - 3)); }
  int8_t col_upper_bound() const { return int8_t(std::min(loc_.col + 4, kBoardDimension)); }

  const Location loc_;
  uint32_t mask_ = 0;
};

}  // namespace detail

piece_orientation_corner_index_t BitBoard::find(Location loc) const {
  util::debug_assert(get(loc));
  util::debug_assert(loc.row == 0 || !get(loc.row - 1, loc.col));  // S neighbor not set
  util::debug_assert(loc.col == 0 || !get(loc.row, loc.col - 1));  // W neighbor not set

  detail::MiniBoard visited(loc);
  detail::MiniBoard queue(loc);
  detail::MiniBoard connected(loc);

  while (queue.any()) {
    Location loc = queue.pop();

    Location neighbors[4] = {
      {int8_t(loc.row - 1), loc.col},
      {int8_t(loc.row + 1), loc.col},
      {loc.row, int8_t(loc.col - 1)},
      {loc.row, int8_t(loc.col + 1)},
    };

    for (int i = 0; i < 4; ++i) {
      Location neighbor = neighbors[i];
      if (!queue.in_bounds(neighbor) || visited.get(neighbor)) continue;

      visited.set(neighbor);
      if (get(neighbor)) {
        queue.set(neighbor);
        connected.set(neighbor);
      }
    }
  }

  return connected.to_piece_orientation_corner_index();
}

}  // namespace blokus
