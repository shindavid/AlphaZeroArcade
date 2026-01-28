#pragma once

#include "core/ActionResponse.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <memory>

namespace search {

template <search::concepts::Traits Traits>
struct AuxData {
  using VerboseData = Traits::VerboseData;
  using VerboseData_sptr = std::shared_ptr<VerboseData>;

  core::ActionResponse action_response;
  VerboseData_sptr verbose_data;
};

}  // namespace search
