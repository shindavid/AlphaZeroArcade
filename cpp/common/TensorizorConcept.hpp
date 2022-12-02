#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>

#include <common/AbstractSymmetryTransform.hpp>
#include <common/BasicTypes.hpp>
#include <util/CppUtil.hpp>

namespace common {

/*
 * All Tensorizor classes must satisfy the TensorizorConcept concept.
 *
 * A Tensorizor is responsible for converting a GameState into a Tensor.
 */
template <class Tensorizor, class GameState>
concept TensorizorConcept = requires(Tensorizor tensorizor, GameState state, typename Tensorizor::InputTensor input) {
  /*
   * The shape of the tensor representation of a game state.
   */
  { typename Tensorizor::Shape{} } -> util::IntSequenceConcept;

  /*
   * Used to clear state between games. (Is this necessary?)
   */
  { tensorizor.clear() };

  /*
   * Receive broadcast of a game state change.
   */
  { tensorizor.receive_state_change(state, action_index_t()) };

  /*
   * Takes an InputTensor reference and writes to it.
   */
  { tensorizor.tensorize(input, GameState{}) };

  { tensorizor.get_random_symmetry(state) } -> util::is_pointer_derived_from<AbstractSymmetryTransform<GameState, Tensorizor>>;
};

}  // namespace common
