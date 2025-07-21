#pragma once

#include "core/BasicTypes.hpp"
#include "core/GameTypes.hpp"
#include "core/TrainingTargets.hpp"
#include "core/concepts/GameConstants.hpp"
#include "core/concepts/GameIO.hpp"
#include "core/concepts/GameInputTensorizor.hpp"
#include "core/concepts/GameMctsConfiguration.hpp"
#include "core/concepts/GameRules.hpp"
#include "core/concepts/GameStateHistory.hpp"
#include "core/concepts/GameSymmetries.hpp"
#include "core/concepts/GameTrainingTargets.hpp"
#include "util/CppUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include "util/MetaProgramming.hpp"

#include <Eigen/Core>

#include <concepts>

namespace core {

namespace concepts {

/*
 * All Game classes G must satisfy core::concepts::Game<G>.
 *
 * Overview of requirements:
 *
 * - G::Constants must be a struct satisfying core::concepts::GameConstants.
 *
 * - G::State must be a trivially-copyable POD struct representing the game state. The neural
 *   network input is constructed from an array of recent State's.
 *
 * - G::StateHistory must be a class that stores a history of recent State's. This is used for
 *   rules-calculations.
 *
 *   For simple games, one can use core::SimpleStateHistory, which assumes that the game rules only
 *   care about the current state, and not the history of states. In a game like chess, however, the
 *   threefold repetition rule and the fifty-move rule require a more sophisticated history.
 *
 * - G::TransformList must be an mp::TypeList that encodes the symmetries of the game. In the game
 *   of go, for instance, since there are 8 symmetries, G::TransformList would contain 8 transform
 *   classes.
 *
 * - G::Rules must be a struct containing static methods for rules calculations.
 *
 * - G::IO must be a struct containing static methods for text input/output.
 *
 * - G::InputTensorizor must be a struct containing a static method for converting an array of
 *   G::State's to a tensor, to be used as input to the neural network. It must also contain
 *   static methods to convert a G::StateHistory to a hashable map-key, to be used for neural
 * network evaluation caching, and for MCGS node reuse.
 *
 * - G::TrainingTargets::List must be an mp::TypeList that encodes the training targets used for
 *   supervised learning. This will include the policy target, the value target, and any other
 *   auxiliary targets.
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
  requires core::concepts::GameInputTensorizor<typename G::InputTensorizor, typename G::State,
                                               typename G::StateHistory>;
  requires core::concepts::GameTrainingTargets<typename G::TrainingTargets, typename G::Types>;

  // Any game-specific one-time static-initialization code should be placed in a static method
  // called static_init().
  {G::static_init()};
};

template <class G>
concept RequiresMctsDoublePass = requires {
  requires core::concepts::Game<G>;
  requires !OperatesOn<typename G::Symmetries, typename G::StateHistory>;
};

}  // namespace concepts

}  // namespace core
