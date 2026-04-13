#pragma once

#include "core/ActionResponse.hpp"
#include "search/concepts/SpecConcept.hpp"

#include <memory>

namespace search {

template <search::concepts::Spec Spec>
struct AuxData {
  using Game = Spec::Game;
  using ActionResponse = core::ActionResponse<Game>;
  using VerboseData = Spec::VerboseData;
  using VerboseData_sptr = std::shared_ptr<VerboseData>;

  ActionResponse action_response;
  VerboseData_sptr verbose_data;
};

}  // namespace search
