#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "core/concepts/Game.hpp"
#include "search/concepts/EdgeConcept.hpp"
#include "search/concepts/NodeConcept.hpp"

namespace search {
namespace concepts {

// Everything in Traits needed for using graph constructs
template <class GT>
concept GraphTraits = requires {
  requires core::concepts::Game<typename GT::Game>;
  requires core::concepts::EvalSpec<typename GT::EvalSpec>;
  requires search::concepts::Node<typename GT::Node, typename GT::EvalSpec>;
  requires search::concepts::Edge<typename GT::Edge>;
};

}  // namespace concepts
}  // namespace search
