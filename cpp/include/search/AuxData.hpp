#pragma once

#include "core/ActionResponse.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <memory>

namespace search {

template <search::concepts::Traits Traits>
struct AuxData {
  using Game = Traits::Game;
  using ActionResponse = core::ActionResponse<Game>;
  using VerboseData = Traits::VerboseData;
  using VerboseData_sptr = std::shared_ptr<VerboseData>;

  ActionResponse action_response;
  VerboseData_sptr verbose_data;
};

}  // namespace search
