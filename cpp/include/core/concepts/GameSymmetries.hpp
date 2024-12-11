#pragma once

#include <core/BasicTypes.hpp>
#include <util/CppUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename GS, typename T>
concept OperatesOn = requires(util::strict_type_match_t<T&> t, group::element_t sym) {
  { GS::apply(t, sym) };
};

template <typename GS, typename T>
concept OperatesOnWithActionType =
    requires(util::strict_type_match_t<T&> t, group::element_t sym, action_mode_t action_mode) {
      { GS::apply(t, action_mode, sym) };
    };

/*
 * One may optionally provide an addition static function:
 *
 * void apply(StateHistory&, group::element_t);
 *
 * Specifying this allows for MCTS optimizations. Without specifying this, MCTS must resort to a
 * "double-pass" through the game-tree on each tree-traversal. We typically expect this overhead to
 * be small compared to the greater cost of neural network inference, but conceivably for some
 * games it could be wise to avoid this if possible.
 */
template <typename GS, typename GameTypes, typename State>
concept GameSymmetries = requires(const State& state) {
  { GS::get_mask(state) } -> std::same_as<typename GameTypes::SymmetryMask>;
  requires core::concepts::OperatesOn<GS, State>;
  requires core::concepts::OperatesOnWithActionType<GS, typename GameTypes::PolicyTensor>;
  requires core::concepts::OperatesOnWithActionType<GS, core::action_t>;
  { GS::get_canonical_symmetry(state) } -> std::same_as<group::element_t>;
};

}  // namespace concepts
}  // namespace core
