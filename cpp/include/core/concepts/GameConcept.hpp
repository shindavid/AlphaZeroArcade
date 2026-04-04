#pragma once

#include "core/GameTypes.hpp"
#include "core/concepts/GameConstantsConcept.hpp"
#include "core/concepts/GameIOConcept.hpp"
#include "core/concepts/GameRulesConcept.hpp"
#include "core/concepts/MoveConcept.hpp"
#include "core/concepts/MoveListConcept.hpp"
#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core {

namespace concepts {

/*
 * All Game classes G must satisfy core::concepts::Game<G>.
 */
template <class G>
concept Game = requires {
  requires core::concepts::GameConstants<typename G::Constants>;
  requires std::same_as<
    typename G::Types,
    core::GameTypes<typename G::Constants, typename G::Move, typename G::MoveList,
                    typename G::State, typename G::GameResults, typename G::SymmetryGroup>>;

  requires std::is_default_constructible_v<typename G::State>;

  requires core::concepts::Move<typename G::Move, typename G::State>;
  requires core::concepts::MoveList<typename G::MoveList, typename G::Move>;

  requires group::concepts::FiniteGroup<typename G::SymmetryGroup>;
  requires core::concepts::GameRules<typename G::Rules, typename G::Types, typename G::State,
                                     typename G::Move>;
  requires core::concepts::GameIO<typename G::IO, typename G::Move, typename G::Types>;

  // Any game-specific one-time static-initialization code should be placed in a static method
  // called static_init().
  { G::static_init() };
};

}  // namespace concepts

}  // namespace core
