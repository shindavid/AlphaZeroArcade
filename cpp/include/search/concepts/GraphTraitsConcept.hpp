#pragma once

#include <concepts>

namespace search::concepts {

template <typename T>
concept GraphTraits = requires {
  typename T::Game;
  typename T::Node;
  typename T::Edge;
  typename T::Move;
  typename T::MoveSet;
  typename T::TransposeKey;

  requires std::same_as<typename T::Move, typename T::Game::Move>;
  requires std::same_as<typename T::MoveSet, typename T::Game::MoveSet>;
};

}  // namespace search::concepts
