#pragma once

#include <bitset>
#include <concepts>
#include <cstdint>
#include <type_traits>

#include <common/AbstractSymmetryTransform.hpp>
#include <common/BasicTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/EigenUtil.hpp>

namespace common {

/*
 * All Tensorizor classes must satisfy the TensorizorConcept concept.
 *
 * A Tensorizor is responsible for converting a GameState into a Tensor.
 */
template <class Tensorizor, class GameState>
concept TensorizorConcept = requires(Tensorizor tensorizor, typename Tensorizor::InputTensor input)
{
  /*
   * The Tensor type used to represent the game state.
   */
  { typename Tensorizor::InputTensor{} } -> eigen_util::FixedTensorConcept;

  /*
   * The maximum number of symmetries.
   */
  { util::decay_copy(Tensorizor::kMaxNumSymmetries) } -> std::same_as<int>;

  /*
   * Used to clear state between games. (Is this necessary?)
   */
  { tensorizor.clear() };

  /*
   * Receive broadcast of a game state change.
   */
  { tensorizor.receive_state_change(GameState{}, action_index_t{}) };

  /*
   * Takes an eigen Tensor reference and writes to it.
   */
  { tensorizor.tensorize(input, GameState{}) };

  { tensorizor.get_symmetry_indices(GameState{}) } -> std::same_as<std::bitset<Tensorizor::kMaxNumSymmetries>>;

  { tensorizor.get_symmetry(symmetry_index_t{}) } -> util::is_pointer_derived_from<AbstractSymmetryTransform<GameState, Tensorizor>>;
};

}  // namespace common
