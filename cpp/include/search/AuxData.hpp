#pragma once

#include "alpha0/VerboseData.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "core/ActionResponse.hpp"

#include <memory>

namespace search {

template <::alpha0::concepts::Spec Spec>
struct AuxData {
  using Game = Spec::Game;
  using ActionResponse = core::ActionResponse<Game>;
  using VerboseData = alpha0::VerboseData<Spec>;
  using VerboseData_sptr = std::shared_ptr<VerboseData>;

  ActionResponse action_response;
  VerboseData_sptr verbose_data;
};

}  // namespace search
