#pragma once

#include "core/Constants.hpp"
#include "core/concepts/GameConcept.hpp"

namespace core {

template <core::concepts::Game Game, SearchParadigm Paradigm>
struct EvalSpec {};  // no definition: require a specialization per game and paradigm

}  // namespace core
