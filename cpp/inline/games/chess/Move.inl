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

inline size_t MoveList::serialize(char* buffer) const {
  // TODO: the plan is to remove MoveList from the ActionPrompt payload, and have GameServerProxy
  // compute legal moves instead. Then, serialize()/deserialize() can be removed from the MoveList
  // interface.
  throw util::Exception("a0achess::MoveList::serialize() not implemented");
}

inline size_t MoveList::deserialize(const char* buffer) {
  // TODO: the plan is to remove MoveList from the ActionPrompt payload, and have GameServerProxy
  // compute legal moves instead. Then, serialize()/deserialize() can be removed from the MoveList
  // interface.
  throw util::Exception("a0achess::MoveList::deserialize() not implemented");
}

inline std::string MoveList::to_string() const {
  // TODO: to_string() is currently only used in x0::SearchResults::to_json(), which is only used
  // for goldenfile testing. We plan to just remove the inclusion of legal moves from the
  // SearchResults json. Then, to_string() can be removed from the MoveList interface.
  throw util::Exception("a0achess::MoveList::to_string() not implemented");
}

}  // namespace a0achess
