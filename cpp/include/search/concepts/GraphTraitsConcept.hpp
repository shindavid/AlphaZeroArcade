#pragma once

#include "core/concepts/Game.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/NodeConcept.hpp"

namespace search {
namespace concepts {

template <class GT>
concept GraphTraits = requires {
  requires core::concepts::Game<typename GT::Game>;
  requires search::concepts::Node<typename GT::Node, typename GT::Game>;
  requires search::concepts::Edge<typename GT::Edge>;
};

}  // namespace concepts
}  // namespace search
