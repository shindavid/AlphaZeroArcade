#include <games/blokus/Types.hpp>

#include <util/Asserts.hpp>

#include <algorithm>
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
  MiniBoard(Location loc)
      : loc_(loc),
        row_bounds_{loc.row, int8_t(std::min(loc.row + 5, kBoardDimension))},
        col_bounds_{int8_t(std::max(0, loc.col - 3)),
                    int8_t(std::min(loc.col + 5, kBoardDimension))} {}

  bool any() const { return mask_; }
  uint32_t mask() const { return mask_; }

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
    int r = (i + 3) / 8;
    int c = (i + r + 2 + (i < 5)) % 8 - 3;
    return {int8_t(loc_.row + r), int8_t(loc_.col + c)};
  }

  int location_to_index(Location loc) const {
    int r = loc.row - loc_.row;
    int c = loc.col - loc_.col;
    return 7 * r + c + 1 - (r < 1);
  }

  int8_t row_lower_bound() const { return row_bounds_[0]; }
  int8_t row_upper_bound() const { return row_bounds_[1]; }
  int8_t col_lower_bound() const { return col_bounds_[0]; }
  int8_t col_upper_bound() const { return col_bounds_[1]; }

  uint32_t mask_ = 1;
  const Location loc_;
  const int8_t row_bounds_[2];
  const int8_t col_bounds_[2];
};

void find_helper(const BitBoard* board, Location loc, int8_t dr, int8_t dc, MiniBoard& visited,
                 MiniBoard& queue, MiniBoard& connected) {
  Location neighbor = loc;
  neighbor.row += dr;
  neighbor.col += dc;

  if (!visited.in_bounds(neighbor) || visited.get(neighbor)) return;
  visited.set(neighbor);
  if (board->get(neighbor)) {
    queue.set(neighbor);
    connected.set(neighbor);
  }
}

}  // namespace detail

std::string BitBoard::to_string(drawing_t c) const {
  BoardString board_string;
  board_string.set(*this, c);
  std::ostringstream ss;
  board_string.print(ss);
  return ss.str();
}

piece_orientation_corner_index_t BitBoard::find(Location loc) const {
  util::debug_assert(get(loc));
  util::debug_assert(loc.row == 0 || !get(loc.row - 1, loc.col));  // S neighbor not set
  util::debug_assert(loc.col == 0 || !get(loc.row, loc.col - 1));  // W neighbor not set

  detail::MiniBoard visited(loc);
  detail::MiniBoard queue(loc);
  detail::MiniBoard connected(loc);

  while (queue.any()) {
    loc = queue.pop();

    detail::find_helper(this, loc, +1, 0, visited, queue, connected);
    detail::find_helper(this, loc, 0, +1, visited, queue, connected);
    detail::find_helper(this, loc, -1, 0, visited, queue, connected);
    detail::find_helper(this, loc, 0, -1, visited, queue, connected);
  }

  return connected.to_piece_orientation_corner_index();
}

void BoardString::print(std::ostream& os, bool omit_trivial_rows) const {
  constexpr char chars[dNumDrawings] = {'.', 'B', 'Y', 'R', 'G', 'o', '+', '*', 'x'};

  os << "   ";
  for (int col = 0; col < kBoardDimension; ++col) {
    os << static_cast<char>('A' + col);
  }
  os << '\n';
  for (int row = kBoardDimension - 1; row >= 0; --row) {
    if (omit_trivial_rows) {
      bool trivial = true;
      for (int col = 0; col < kBoardDimension; ++col) {
        if (colors_[row][col] != dBlankSpace) {
          trivial = false;
          break;
        }
      }
      if (trivial) continue;
    }

    os << std::setw(2) << (row + 1) << ' ';
    for (int col = 0; col < kBoardDimension; ++col) {
      os << chars[colors_[row][col]];
    }
    os << ' ' << std::setw(2) << (row + 1) << '\n';
  }
  os << "   ";
  for (int col = 0; col < kBoardDimension; ++col) {
    os << static_cast<char>('A' + col);
  }
  os << '\n';
}

void BoardString::pretty_print(std::ostream& os) const {
  if (!util::tty_mode()) {
    print(os);
    return;
  }

  constexpr int N = 16384;
  char buffer[N] = "";
  int c = 0;

  constexpr const char* color_strs[dNumDrawings] = {
      "  ",                         // dBlankSpace
      "\033[44m  \033[0m",          // dBlueSpace
      "\033[43m  \033[0m",          // dYellowSpace
      "\033[41m  \033[0m",          // dRedSpace
      "\033[42m  \033[0m",          // dGreenSpace
      "\033[47m\033[34m⚫\033[0m",  // dCircle
      "\033[47m\033[34m+\033[0m",   // dPlus
      "\033[47m\033[34m*\033[0m",   // dStar
      "\033[47m\033[34m×\033[0m"    // dTimes
  };

  c += sprintf(buffer + c, "   ");
  for (int col = 0; col < kBoardDimension; ++col) {
    c += sprintf(buffer + c, " %c", 'A' + col);
  }
  c += sprintf(buffer + c, "\n");

  for (int row = kBoardDimension - 1; row >= 0; --row) {
    c += sprintf(buffer + c, "%2d ", row + 1);
    for (int col = 0; col < kBoardDimension; ++col) {
      drawing_t d = colors_[row][col];
      util::debug_assert(d >= 0 && d < 5, "%d", int(d));
      c += snprintf(buffer + c, std::max(N - c, 0), "%s", color_strs[d]);
    }
    c += sprintf(buffer + c, " %2d\n", row + 1);
  }

  c += sprintf(buffer + c, "   ");
  for (int col = 0; col < kBoardDimension; ++col) {
    c += sprintf(buffer + c, " %c", 'A' + col);
  }
  c += sprintf(buffer + c, "\n");

  util::release_assert(c < N, "BoardString::pretty_print() overflow (%d < %d)", c, N);
  os << buffer;
}

void TuiPrompt::print() {
  int width = util::get_screen_width();

  std::ostringstream print_lines[kNumLines];

  int cur_width = 0;
  for (const block_t& block : blocks_) {
    int potential_width = cur_width + block.width;
    if (potential_width >= width) {
      for (auto& line : print_lines) {
        std::cout << line.str() << std::endl;
      }
      std::cout << std::endl;
      cur_width = 0;
    }

    for (int i = 0; i < kNumLines; ++i) {
      print_lines[i] << block.lines[kNumLines - i - 1].str();
    }
    cur_width += block.width;
  }

  for (const auto& line : print_lines) {
    std::cout << line.str() << std::endl;
  }
  std::cout << std::endl;
}

void Piece::write_to(TuiPrompt& prompt, color_t color) const {
  PieceOrientation canonical = canonical_orientation();
  canonical.write_to(prompt, color, index_);
}

void PieceOrientation::write_to(TuiPrompt& prompt, color_t color, int label) const {
  std::string label_str = util::create_string("%d", label);

  const uint8_t* row_masks = this->row_masks();
  int height = this->height();
  int width = this->width();

  const char* reset = "\033[0m";

  constexpr const char* color_strs[kNumColors] = {
      "\033[44m",  // cBlue
      "\033[43m",  // cYellow
      "\033[41m",  // cRed
      "\033[42m"   // cGreen
  };
  const char* color_str = color_strs[color];

  TuiPrompt::block_t& block = prompt.blocks_.emplace_back();
  block.width = std::max((int)label_str.size(), 2 * width) + 1;
  for (int r = 0; r < height; ++r) {
    std::ostringstream& line = block.lines[r + 2];
    bool cur_set = false;
    uint8_t mask = row_masks[r];
    for (int c = 0; c < width; ++c) {
      bool set = mask & (1 << (c + 1));
      if (cur_set && !set) {
        line << reset;
      } else if (!cur_set && set) {
        line << color_str;
      }
      cur_set = set;
      line << "  ";
    }

    if (cur_set) {
      line << reset;
    }

    for (int k = 0; k < (block.width - 2 * width); ++k) {
      line << " ";
    }
  }

  for (int r = height; r < 5; ++r) {
    std::ostringstream& line = block.lines[r + 2];
    for (int k = 0; k < block.width; ++k) {
      line << " ";
    }
  }

  block.lines[0] << label_str;
  for (int k = 0; k < block.width - (int)label_str.size(); ++k) {
    block.lines[0] << " ";
  }
}

void PieceOrientationCorner::pretty_print(std::ostream& os, color_t color) const {
  PieceOrientation po = to_piece_orientation();
  const uint8_t* row_masks = po.row_masks();
  int height = po.height();
  int width = po.width();

  Location corner = corner_offset();
  int corner_r = corner.row - 1;
  int corner_c = corner.col - 1;

  // printf("DBG corner:%s (%d)\n", corner.to_string().c_str(), index_);
  // std::cout.flush();

  const char* reset = "\033[0m";

  constexpr const char* color_strs[kNumColors] = {
      "\033[44m",  // cBlue
      "\033[43m",  // cYellow
      "\033[41m",  // cRed
      "\033[42m"   // cGreen
  };
  const char* color_str = color_strs[color];

  std::ostringstream streams[height];

  for (int r = 0; r < height; ++r) {
    std::ostringstream& stream = streams[r];
    bool cur_set = false;
    uint8_t mask = row_masks[r];
    for (int c = 0; c < width; ++c) {
      bool set = mask & (1 << (c + 1));
      if (cur_set && !set) {
        stream << reset;
      } else if (!cur_set && set) {
        stream << color_str;
      }
      cur_set = set;
      bool blink = r == corner_r && c == corner_c;
      if (blink) {
        stream << ansi::kBlink() << "><" << reset;
        cur_set = false;
      } else {
        stream << "  ";
      }
    }

    if (cur_set) {
      stream << reset;
    }
  }

  for (int r = height - 1; r >= 0; --r) {
    os << streams[r].str() << std::endl;
  }
}

}  // namespace blokus
