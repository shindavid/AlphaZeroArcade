#pragma once

#include "core/ActionResponse.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

#include <memory>

namespace search {

template <search::concepts::SearchSpec SearchSpec>
struct AuxData {
  using Game = SearchSpec::Game;
  using ActionResponse = core::ActionResponse<Game>;
  using VerboseData = SearchSpec::VerboseData;
  using VerboseData_sptr = std::shared_ptr<VerboseData>;

  ActionResponse action_response;
  VerboseData_sptr verbose_data;
};

}  // namespace search
