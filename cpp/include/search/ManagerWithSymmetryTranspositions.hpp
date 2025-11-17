#include "search/ManagerBase.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
class ManagerWithSymmetryTranspositions
    : public ManagerBase<Traits, ManagerWithSymmetryTranspositions<Traits>> {
 public:
  using Base = ManagerBase<Traits, ManagerWithSymmetryTranspositions<Traits>>;
  using Base::Base;
};

}  // namespace search
