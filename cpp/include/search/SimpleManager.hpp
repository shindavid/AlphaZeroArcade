#include "search/ManagerBase.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
class SimpleManager : public ManagerBase<Traits, SimpleManager<Traits>> {
 public:
  using Base = ManagerBase<Traits, SimpleManager<Traits>>;
  using Base::Base;

 protected:
  void update(core::action_t);
};

}  // namespace search
