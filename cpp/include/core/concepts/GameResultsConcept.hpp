#pragma once

#include "core/BasicTypes.hpp"
#include "util/EigenUtil.hpp"

#include <concepts>

namespace core {
namespace concepts {

/*
 * A class satisfying the GameResults concept must provide various type declarations and static
 * methods involving representations of the results of a game.
 *
 * Two type declarations must be provided: Tensor and ValueArray.
 *
 * - Tensor: this representation is what the game rules outputs, and what the value-head of the
 *   neural network aims to learn.
 *
 * - ValueArray: this representation is what is used during MCTS. This type must be an array of
 *   length Games::Constants::kNumPlayers.
 */
template <class GR>
concept GameResults = requires(
  const typename GR::Tensor& const_tensor_ref, typename GR::Tensor& tensor_ref,
  const typename GR::ValueArray& win_rates, const eigen_util::PrintArrayFormatMap* fmt_map) {
  // The maximum/minimum value that can be present in the ValueArray representation. This is used
  // for provably-winning/losing checks, and also for virtual-loss calculations.
  { util::decay_copy(GR::kMinValue) } -> std::same_as<float>;
  { util::decay_copy(GR::kMaxValue) } -> std::same_as<float>;

  /*
   * The tensor representation of the game result. This is what the game rules outputs, and what the
   * value-head of the neural network aims to learn.
   *
   * In general, this representation may not be equivalent to the ValueArray representation. For
   * instance, in a 2-player game that allows draws, the tensor may have 3 elements (W, L, D). The
   * array representation would instead have 2 elements (W', L'), where W' = W + 0.5*D and
   * L' = L + 0.5*D (this assumes that a draw is half as valuable as a win).
   */
  requires eigen_util::concepts::FTensor<typename GR::Tensor>;

  /*
   * The array representation of the game result. This is what is used during MCTS. This type must
   * be an array of length Games::Constants::kNumPlayers.
   */
  requires eigen_util::concepts::FArray<typename GR::ValueArray>;

  /*
   * Create a tensor representation of a win for the given player.
   *
   * This method is needed to support the --respect-victory-hints flag in a generic way.
   */
  { GR::win(core::seat_index_t{}) } -> std::same_as<typename GR::Tensor>;

  /*
   * Convert the tensor representation to the array representation.
   *
   * If this is a game that allows draws, this function is where assumptions about the relative
   * value of draws and wins are encoded.
   */
  { GR::to_value_array(const_tensor_ref) } -> std::same_as<typename GR::ValueArray>;

  /*
   * Modify the tensor representation to reflect a left rotation of the players.
   */
  { GR::left_rotate(tensor_ref, core::seat_index_t{}) };

  /*
   * Modify the tensor representation to reflect a right rotation of the players.
   */
  { GR::right_rotate(tensor_ref, core::seat_index_t{}) };

  /*
   * Print the V (from the neural net) and Q (from search updates) side by side.
   */
  { GR::print_array(const_tensor_ref, win_rates, fmt_map) };
};

}  // namespace concepts
}  // namespace core
