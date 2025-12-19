#pragma once

#include "search/concepts/ManagerParamsConcept.hpp"

namespace beta0 {

template <search::concepts::ManagerParams ManagerParams>
struct AuxState {
  AuxState(const ManagerParams& params) {}

  void clear() {}
  void step() {}
};

}  // namespace beta0
