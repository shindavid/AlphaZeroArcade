#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/ManagerParamsBase.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
struct ManagerParams : public search::ManagerParamsBase<EvalSpec> {
  using Base = search::ManagerParamsBase<EvalSpec>;

  ManagerParams(search::Mode);

  bool operator==(const ManagerParams& other) const = default;

  /*
   * For dirichlet noise, we use a uniform alpha = dirichlet_alpha_factor / sqrt(num_actions).
   */
  float dirichlet_alpha_factor = 0.57;
  bool enable_exploratory_visits = false;
};

}  // namespace beta0

#include "inline/beta0/ManagerParams.inl"
