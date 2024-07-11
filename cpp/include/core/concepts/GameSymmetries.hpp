#pragma once

#include <core/BasicTypes.hpp>
#include <util/FiniteGroups.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename GS, typename GameTypes, typename BaseState>
concept GameSymmetries = requires(const BaseState& const_base_state, BaseState& base_state,
                                  action_t& action, typename GameTypes::PolicyTensor& policy,
                                  group::element_t& sym) {
  { GS::get_mask(const_base_state) } -> std::same_as<typename GameTypes::SymmetryMask>;
  { GS::apply(base_state, sym) };
  { GS::apply(policy, sym) };
  { GS::apply(action, sym) };
  { GS::get_canonical_symmetry(const_base_state) } -> std::same_as<group::element_t>;
};

}  // namespace concepts
}  // namespace core
