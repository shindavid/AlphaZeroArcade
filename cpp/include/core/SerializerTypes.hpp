#pragma once

#include <core/concepts/Game.hpp>
#include <core/serializers/GeneralSerializer.hpp>

namespace core {

/*
 * serializer_t is a metafunction that maps a GameState to the appropriate serializer type. By
 * default, this is GeneralSerializer<GameState>, but it can be specialized for specific games.
 */
template <concepts::Game Game>
struct serializer {
  using type = GeneralSerializer<GameState>;
};
template <concepts::Game Game>
using serializer_t = typename serializer<GameState>::type;

}  // namespace core
