#include "games/blokus/Move.hpp"

#include "games/blokus/Constants.hpp"
#include "games/blokus/Types.hpp"
#include "util/Exceptions.hpp"
#include "util/StringUtil.hpp"

#include <format>

namespace blokus {

inline std::string Move::to_str() const {
  if (phase_ == blokus::kLocationPhase) {
    if (*this == pass()) {
      return "pass";
    }
    Location loc = Location::unflatten(index_);
    int row = loc.row + 1;
    char col = 'A' + loc.col;
    return std::format("{}{}", col, row);
  } else if (phase_ == blokus::kPiecePlacementPhase) {
    PieceOrientationCorner poc = PieceOrientationCorner(index_);
    return poc.name();
  } else {
    throw util::Exception("invalid move phase: {}", phase_);
  }
}

inline Move Move::from_str(const GameState&, std::string_view s) {
  if (s == "pass") {
    return pass();
  }

  if (s.empty()) {
    throw util::Exception("invalid move string: {}", s);
  }

  char c = s[0];
  int letter_index = c - 'A';
  if (letter_index >= 0 && letter_index < kBoardDimension) {
    int row = util::atoi(s.substr(1)) - 1;
    int col = letter_index;
    Location loc{row, col};
    return Move(loc.flatten(), blokus::kLocationPhase);
  }

  try {
    int index = util::atoi(s);
    PieceOrientationCorner poc = PieceOrientationCorner(index);
    return Move(poc.index(), blokus::kPiecePlacementPhase);
  } catch (const std::exception&) {
    throw util::Exception("invalid move string: {}", s);
  }
}

inline std::string Move::serialize() const { return std::format("{}.{}", index_, phase_); }

inline Move Move::deserialize(std::string_view s) {
  size_t dot_pos = s.find('.');
  if (dot_pos == std::string_view::npos) {
    throw util::Exception("invalid move string: {}", s);
  }
  int16_t index = util::atoi(s.substr(0, dot_pos));
  core::game_phase_t phase = util::atoi(s.substr(dot_pos + 1));
  return Move(index, phase);
}

}  // namespace blokus
