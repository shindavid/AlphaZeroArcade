
#pragma once

#include "search/concepts/AuxStateConcept.hpp"
#include "search/concepts/GraphTraitsConcept.hpp"
#include "search/concepts/ManagerParamsConcept.hpp"

namespace search {
namespace concepts {

// Everything in Traits needed for search::GeneralContext
template <class GCT>
concept GeneralContextTraits = requires {
  requires search::concepts::GraphTraits<GCT>;
  requires search::concepts::ManagerParams<typename GCT::ManagerParams>;
  requires search::concepts::AuxState<typename GCT::AuxState, typename GCT::ManagerParams>;
};

}  // namespace concepts
}  // namespace search
