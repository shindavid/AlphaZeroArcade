#include <games/blokus/Types.hpp>

#include <util/Asserts.hpp>

#include <bit>

namespace blokus {

namespace detail {

// MiniBoard is a specialized struct that is just large enough to store the location of a single
// piece on the board.
class MiniBoard {
 public:
  MiniBoard(Location loc) {
    loc_ = loc;
    rows_[0] = 1 << 3;
    for (int r = 1; r < 5; ++r) {
      rows_[r] = 0;
    }
  }

  bool any() const {
    for (int r = 0; r < 5; ++r) {
      if (rows_[r]) return true;
    }
    return false;
  }

  Location pop() {
    for (int r = 0; r < 5; ++r) {
      if (rows_[r]) {
        int index = std::countr_zero(rows_[r]);
        rows_[r] &= ~(1 << index);
        return {int8_t(r + loc_.row), int8_t(index + loc_.col - 3)};
      }
    }

    throw std::runtime_error("MiniBoard::pop() called on empty MiniBoard");
  }

  void set(Location loc) {
    loc = calibrate(loc);
    rows_[loc.row] |= 1 << loc.col;
  }

  bool get(Location loc) const {
    loc = calibrate(loc);
    return rows_[loc.row] & (1 << loc.col);
  }

  bool in_bounds(Location loc) const {
    int8_t r = loc.row - loc_.row;
    int8_t c = loc.col - loc_.col;
    return 0 <= r && r < 5 && 0 <= c && c < 5;
  }

  piece_orientation_corner_index_t to_piece_orientation_corner_index() const {
    throw std::runtime_error("MiniBoard::to_piece_orientation_corner_index() not implemented");
  }

 private:
  Location calibrate(Location loc) const {
    int8_t r = loc.row - loc_.row;
    int8_t c = loc.col - loc_.col + 3;
    util::debug_assert(0 <= r && r < 5);
    util::debug_assert(0 <= c && c < 5);
    return {r, c};
  }

  Location loc_;
  uint8_t rows_[5];
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
