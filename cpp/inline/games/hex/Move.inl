#include "games/hex/Move.hpp"

#include "util/StringUtil.hpp"

#include <format>

namespace hex {

inline std::string Move::to_str() const {
  if (*this == swap()) {
    return "swap";
  } else {
    return std::format("{:c}{}", 'A' + col_, row_ + 1);
  }
}

inline Move Move::from_str(std::string_view s) {
  if (s == "swap") {
    return swap();
  } else {
    if (s.size() < 2) throw std::invalid_argument("invalid move string");
    char col_c = s[0];
    int row = util::atoi(s.substr(1)) - 1;
    int col = col_c - 'A';
    return Move(row, col);
  }
}

inline std::string Move::serialize() const { return std::format("{}", vertex()); }

inline Move Move::deserialize(std::string_view s) {
  return Move(util::atoi(s));
}

}  // namespace hex
