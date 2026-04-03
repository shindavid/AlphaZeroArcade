#include "games/stochastic_nim/Move.hpp"

#include "games/stochastic_nim/Constants.hpp"
#include "util/Exceptions.hpp"
#include "util/StringUtil.hpp"

#include <format>

namespace stochastic_nim {

inline std::string Move::to_str() const {
  if (phase_ == stochastic_nim::kChancePhase) {
    return std::format("r{}", index_);
  } else if (phase_ == stochastic_nim::kPlayerPhase) {
    return std::format("{}", index_ + 1);
  } else {
    throw util::Exception("invalid move phase: {}", phase_);
  }
}

inline Move Move::from_str(const GameState&, std::string_view s) {
  char c = s[0];
  if (c == 'r') {
    int index = util::atoi(s.substr(1));
    return Move(index, stochastic_nim::kChancePhase);
  } else {
    int index = util::atoi(s);
    return Move(index, stochastic_nim::kPlayerPhase);
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

}  // namespace stochastic_nim
