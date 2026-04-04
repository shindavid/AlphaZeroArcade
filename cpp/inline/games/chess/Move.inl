#include "games/chess/Move.hpp"

#include "util/Exceptions.hpp"

namespace a0achess {

inline Move Move::from_str(const GameState& state, std::string_view s) {
  return Move(chess::uci::uciToMove(state, s).move());
}

inline Move MoveList::get_random(std::mt19937& prng) const {  // assumes !empty()
  std::uniform_int_distribution<int> dist(0, count() - 1);
  return (*this)[dist(prng)];
}

}  // namespace a0achess
