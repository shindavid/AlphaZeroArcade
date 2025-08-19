#pragma once

namespace search {

template <typename Traits>
class ManagerBase {
 public:
  using Game = Traits::Game;
  using ActionSelector = Traits::ActionSelector;
  using Node = Traits::Node;
  using SearchContext = Traits::SearchContext;
};

}  // namespace search

#include "inline/search/ManagerBase.inl"
