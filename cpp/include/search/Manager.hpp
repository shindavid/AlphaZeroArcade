#pragma once

#include "core/Constants.hpp"
#include "search/ManagerBase.hpp"
#include "search/concepts/TraitsConcept.hpp"

namespace search {

template <search::concepts::Traits Traits, core::TranspositionRule TranspositionRule>
class ManagerImpl;

template <search::concepts::Traits Traits>
class ManagerImpl<Traits, core::kSimpleTranspositions>
    : public ManagerBase<Traits, ManagerImpl<Traits, core::kSimpleTranspositions>> {
 public:
  static constexpr core::TranspositionRule kTranspositionRule = core::kSimpleTranspositions;
  using Impl = ManagerImpl<Traits, kTranspositionRule>;
  using Base = ManagerBase<Traits, Impl>;
  using Game = Base::Game;
  using Node = Base::Node;
  using State = Base::State;

  using Base::Base;

  void update(core::action_t);
};

template <search::concepts::Traits Traits>
class ManagerImpl<Traits, core::kSymmetryTranspositions>
    : public ManagerBase<Traits, ManagerImpl<Traits, core::kSymmetryTranspositions>> {
 public:
  static constexpr core::TranspositionRule kTranspositionRule = core::kSymmetryTranspositions;
  using Impl = ManagerImpl<Traits, kTranspositionRule>;
  using Base = ManagerBase<Traits, Impl>;
  using Game = Base::Game;
  using Node = Base::Node;
  using State = Base::State;

  using Base::Base;

  void update(core::action_t);
};

template <search::concepts::Traits Traits>
class ManagerImpl<Traits, core::kNoTranspositions>
    : public ManagerBase<Traits, ManagerImpl<Traits, core::kNoTranspositions>> {
 public:
  static constexpr core::TranspositionRule kTranspositionRule = core::kNoTranspositions;
  using Impl = ManagerImpl<Traits, kTranspositionRule>;
  using Base = ManagerBase<Traits, Impl>;
  using Game = Base::Game;
  using Node = Base::Node;
  using State = Base::State;

  using Base::Base;

  void update(core::action_t);
};

template <search::concepts::Traits Traits>
using Manager = ManagerImpl<Traits, Traits::EvalSpec::MctsConfiguration::kTranspositionRule>;

}  // namespace search

#include "inline/search/Manager.inl"
