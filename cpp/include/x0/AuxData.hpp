#pragma once

#include "core/ActionResponse.hpp"

namespace x0 {

struct AuxData {
  core::ActionResponse action_response;

  AuxData(const core::ActionResponse& ar) : action_response(ar) {}
  virtual ~AuxData() = default;
};

}  // namespace x0
