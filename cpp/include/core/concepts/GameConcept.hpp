#pragma once

#include "core/GameTraits.hpp"
#include "core/concepts/GameConstantsConcept.hpp"
#include "core/concepts/GameIOConcept.hpp"
#include "core/concepts/GameRulesConcept.hpp"
#include "core/concepts/MoveConcept.hpp"
#include "core/concepts/MoveSetConcept.hpp"
#include "core/concepts/PlayerResultConcept.hpp"
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
    typename G::Types, core::GameTraits<typename G::Constants, typename G::Move,
                                        typename G::MoveSet, typename G::State, typename G::InfoSet,
                                        typename G::PlayerResult, typename G::SymmetryGroup>>;

  requires std::is_default_constructible_v<typename G::State>;
  requires std::is_default_constructible_v<typename G::InfoSet>;

  requires core::concepts::Move<typename G::Move, typename G::InfoSet>;
  requires core::concepts::MoveSet<typename G::MoveSet, typename G::Move>;
  requires core::concepts::PlayerResult<typename G::PlayerResult>;

  requires group::concepts::FiniteGroup<typename G::SymmetryGroup>;
  requires core::concepts::GameRules<typename G::Rules, typename G::Types, typename G::State,
                                     typename G::InfoSet, typename G::Move>;
  requires core::concepts::GameIO<typename G::IO, typename G::Move, typename G::Types>;
};

}  // namespace concepts

}  // namespace core
