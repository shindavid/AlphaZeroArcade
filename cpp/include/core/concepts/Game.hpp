#pragma once

#include <array>
#include <bitset>
#include <concepts>
#include <string>

#include <Eigen/Core>

#include <core/BasicTypes.hpp>
#include <core/GameTypes.hpp>
#include <core/TrainingTargets.hpp>
#include <core/concepts/GameConstants.hpp>
#include <core/concepts/GameIO.hpp>
#include <core/concepts/GameInputTensorizor.hpp>
#include <core/concepts/GameRules.hpp>
#include <core/concepts/GameSymmetries.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>
#include <util/MetaProgramming.hpp>

namespace core {

namespace concepts {

/*
 * All Game classes G must satisfy core::concepts::Game<G>.
 *
 * Overview of requirements:
 *
 * - G::Constants must be a struct satisfying core::concepts::GameConstants.
 *
 * - G::FullState must be a class fully representing the game state. This must be castable to a
 *   G::BaseState, which should be a trivially-copyable POD struct. The FullState is used for
 *   rules-calculations, while an array of recent BaseState's is used as neural network input.
 *
 *   For simple games, FullState and BaseState can be the same type. In a game like chess, in order
 *   to support the threefold repetition rule, FullState would need to store a history of past
 *   states (or a Zobrist hash set), while BaseState would just store a history-less board state.
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
 *   G::BaseState's to a tensor, to be used as input to the neural network. It must also contain
 *   static methods to convert a G::FullState to a hashable map-key, to be used for neural network
 *   evaluation caching, and for MCGS node reuse.
 *
 * - G::TrainingTargets::List must be an mp::TypeList that encodes the training targets used for
 *   supervised learning. This will include the policy target, the value target, and any other
 *   auxiliary targets.
 */
template <class G>
concept Game = requires {
  requires core::concepts::GameConstants<typename G::Constants>;
  requires std::same_as<typename G::Types, core::GameTypes<typename G::Constants, typename G::BaseState>>;

  requires std::is_default_constructible_v<typename G::BaseState>;
  requires std::is_trivially_copyable_v<typename G::BaseState>;
  requires std::is_convertible_v<typename G::BaseState, typename G::FullState>;

  // TODO: requirements for Symmmetry class

  requires ::concepts::FiniteGroupList<typename G::SymmetryGroups>;
  requires core::concepts::GameSymmetries<typename G::Symmetries, typename G::Types::PolicyTensor,
                                          typename G::BaseState>;
  requires core::concepts::GameRules<typename G::Rules, typename G::Types, typename G::BaseState,
                                     typename G::FullState>;
  requires core::concepts::GameIO<typename G::IO, typename G::Types, typename G::BaseState>;
  requires core::concepts::GameInputTensorizor<typename G::InputTensorizor, typename G::BaseState,
                                               typename G::FullState>;
  requires core::concepts::TrainingTargetList<typename G::TrainingTargets::List,
                                              typename G::Types::GameLogView>;
};

}  // namespace concepts

}  // namespace core
