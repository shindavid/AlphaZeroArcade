#pragma once

#include <core/BasicTypes.hpp>

#include <concepts>

namespace core {
namespace concepts {

template <typename GS, typename PolicyTensor, typename BaseState>
concept GameSymmetries = requires(const BaseState& const_base_state, BaseState& base_state,
                                  PolicyTensor& policy, const core::symmetry_t& sym) {
  { GS::get_group(const_base_state) } -> std::same_as<core::group_id_t>;
  { GS::apply(base_state, sym) };
  { GS::apply(policy, sym) };
};

}  // namespace concepts
}  // namespace core
