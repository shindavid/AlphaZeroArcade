#pragma once

#include "core/BasicTypes.hpp"
#include "util/CppUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <concepts>

namespace core {
namespace concepts {

template <typename GS, typename T>
concept OperatesOn = requires(util::strict_type_match_t<T&> t, group::element_t sym) {
  { GS::apply(t, sym) };
};

template <typename GS, typename T>
concept OperatesOnWithActionMode =
  requires(T& t, group::element_t sym, action_mode_t action_mode) {
    { GS::apply(t, action_mode, sym) };
  };

template <typename GS, typename GameTypes, typename State>
concept GameSymmetries = requires(const State& state) {
  { GS::get_mask(state) } -> std::same_as<typename GameTypes::SymmetryMask>;
  requires core::concepts::OperatesOn<GS, State>;
  requires core::concepts::OperatesOnWithActionMode<GS, typename GameTypes::PolicyTensor>;
  requires core::concepts::OperatesOnWithActionMode<GS, typename GameTypes::ActionValueTensor>;
  requires core::concepts::OperatesOnWithActionMode<GS, core::action_t>;
  { GS::get_canonical_symmetry(state) } -> std::same_as<group::element_t>;
};

}  // namespace concepts
}  // namespace core
