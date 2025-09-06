#pragma once

#include "core/GameTypes.hpp"
#include "core/concepts/GameConstants.hpp"
#include "core/concepts/GameIO.hpp"
#include "core/concepts/GameMctsConfiguration.hpp"
#include "core/concepts/GameRules.hpp"
#include "core/concepts/GameStateHistory.hpp"
#include "core/concepts/GameSymmetries.hpp"
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
  requires core::concepts::GameMctsConfiguration<typename G::MctsConfiguration>;
  requires std::same_as<typename G::Types,
                        core::GameTypes<typename G::Constants, typename G::State,
                                        typename G::GameResults, typename G::SymmetryGroup>>;

  requires std::is_default_constructible_v<typename G::State>;
  requires std::is_trivially_destructible_v<typename G::State>;
  requires core::concepts::GameStateHistory<typename G::StateHistory, typename G::State,
                                            typename G::Rules>;

  requires group::concepts::FiniteGroup<typename G::SymmetryGroup>;
  requires core::concepts::GameSymmetries<typename G::Symmetries, typename G::Types,
                                          typename G::State>;
  requires core::concepts::GameRules<typename G::Rules, typename G::Types,
                                     typename G::GameResults::Tensor, typename G::State,
                                     typename G::StateHistory>;
  requires core::concepts::GameIO<typename G::IO, typename G::Types>;

  // Any game-specific one-time static-initialization code should be placed in a static method
  // called static_init().
  { G::static_init() };
};

template <class G>
concept RequiresMctsDoublePass = requires {
  requires core::concepts::Game<G>;
  requires !OperatesOn<typename G::Symmetries, typename G::StateHistory>;
};

}  // namespace concepts

}  // namespace core
