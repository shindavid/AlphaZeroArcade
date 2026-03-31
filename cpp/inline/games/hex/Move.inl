#include "games/hex/Move.hpp"

#include <format>

namespace hex {

inline std::string Move::to_str() const {
  if (row_ == Constants::kBoardDim && col_ == Constants::kBoardDim) {
    return "swap";
  } else {
    return std::format("{:c}{}", 'A' + col_, row_ + 1);
  }
}

inline Move Move::from_str(std::string_view s) {
  if (s == "swap") {
    return Move(Constants::kBoardDim, Constants::kBoardDim);
  } else {
    if (s.size() < 2) throw std::invalid_argument("invalid move string");
    char col_c = s[0];
    int row = std::stoi(std::string(s.substr(1))) - 1;
    int col = col_c - 'A';
    return Move(row, col);
  }
}

inline std::string Move::serialize() const { return std::format("{}", vertex()); }

inline Move Move::deserialize(std::string_view s) {
  int v;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
  if (ec != std::errc{}) throw std::invalid_argument("bad move");
  return Move(vertex_t(v));
}

}  // namespace hex
