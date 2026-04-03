#include "games/othello/Move.hpp"

#include "util/StringUtil.hpp"

#include <format>

namespace othello {

inline std::string Move::to_str() const {
  if (*this == pass()) {
    return "PA";
  } else {
    return std::format("{:c}{}", 'A' + col_, row_ + 1);
  }
}

inline Move Move::from_str(const GameState&, std::string_view s) {
  if (s == "PA") {
    return pass();
  } else {
    if (s.size() != 2) throw std::invalid_argument("invalid move string");
    int col = s[0] - 'A';
    int row = s[1] - '1';
    return Move(row, col);
  }
}

}  // namespace othello
