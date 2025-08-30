#pragma once

#include "core/concepts/Game.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/NodeConcept.hpp"

namespace search {

template <core::concepts::Game G, search::concepts::Node<G> N, search::concepts::Edge E>
struct GraphTraits {
  using Game = G;
  using Node = N;
  using Edge = E;

  // static_assert(search::concepts::GraphTraits<GraphTraits>);
};

}  // namespace search
