#pragma once

#include "search/ManagerBase.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits>
class ManagerWithSymmetryTranspositions
    : public ManagerBase<Traits, ManagerWithSymmetryTranspositions<Traits>> {
 public:
  using Base = ManagerBase<Traits, ManagerWithSymmetryTranspositions<Traits>>;
  using State = Base::State;
  using Symmetries = Base::Symmetries;
  using Rules = Base::Rules;
  using Node = Base::Node;
  using Base::Base;

  void update(core::action_t);

};

}  // namespace search

#include "inline/search/ManagerWithSymmetryTranspositions.inl"
