#pragma once

#include "core/SearchParadigm.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/CppUtil.hpp"

#include <concepts>

namespace core::concepts {

// Narrow concept: only requires a paradigm tag and a Game type.
// Used by Bindings::SupportedSpecs iteration, FfiMacro dispatch, and PlayerFactory construction.
// For the full set of MCTS-related requirements, see alpha0::concepts::Spec.
template <typename ES>
concept ParadigmSpec = requires {
  { util::decay_copy(ES::kParadigm) } -> std::same_as<core::SearchParadigm>;

  requires core::concepts::Game<typename ES::Game>;
};

}  // namespace core::concepts
