#pragma once

#include "core/concepts/GameConcept.hpp"

namespace core {

template <core::concepts::Game Game>
struct InputTensorizor;  // no definition: require a specialization per game

}  // namespace core
