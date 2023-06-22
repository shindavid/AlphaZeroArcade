#pragma once

#include <core/GameStateConcept.hpp>
#include <core/serializers/GeneralSerializer.hpp>

namespace common {

/*
 * serializer_t is a metafunction that maps a GameState to the appropriate serializer type. By default, this is
 * GeneralSerializer<GameState>, but it can be specialized for specific games.
 */
template <GameStateConcept GameState>
struct serializer {
  using type = GeneralSerializer<GameState>;
};
template <GameStateConcept GameState> using serializer_t = typename serializer<GameState>::type;

}  // namespace common
