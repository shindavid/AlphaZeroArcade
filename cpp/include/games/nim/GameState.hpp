#pragma once

#include <cstddef>
#include <functional>

namespace nim {

struct GameState {
  auto operator<=>(const GameState& other) const = default;
  size_t hash() const;

  int stones_left;
  int current_player;
};

}  // namespace nim

namespace std {

template <>
struct hash<nim::GameState> {
  size_t operator()(const nim::GameState& pos) const { return pos.hash(); }
};

}  // namespace std
