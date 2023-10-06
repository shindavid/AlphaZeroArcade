#pragma once

#include <bitset>
#include <concepts>
#include <cstdint>
#include <type_traits>

#include <core/AbstractSymmetryTransform.hpp>
#include <core/BasicTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace core {

/*
 * All Tensorizor classes must satisfy the TensorizorConcept concept.
 *
 * A Tensorizor is responsible for converting a GameState into a Tensor.
 *
 * AlphaGo includes a history of the past 7 game states in the input tensor. If you want to include history like this,
 * the Tensorizor class is the appropriate place to maintain that state.
 */
template <class Tensorizor, class GameState>
concept TensorizorConcept = requires(Tensorizor tensorizor, typename Tensorizor::InputTensor input)
{
  /*
   * The Tensor type used to represent the game state.
   */
  { typename Tensorizor::InputTensor{} } -> eigen_util::FixedTensorConcept;

  /*
   * Used to clear state between games. Needed if the Tensorizor maintains state, as is done when recent history is
   * included as part of the input tensor.
   */
  { tensorizor.clear() };

  /*
   * Receive broadcast of a game state change.
   */
  { tensorizor.receive_state_change(GameState{}, typename GameState::Action{}) };

  /*
   * Takes an eigen Tensor reference and writes to it.
   */
  { tensorizor.tensorize(input, GameState{}) };
};

}  // namespace core
