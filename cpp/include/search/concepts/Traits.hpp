#pragma once

#include "core/concepts/Game.hpp"
// #include "search/concepts/Node.hpp"

namespace search {
namespace concepts {

template <class T>
concept Traits = requires {
  requires core::concepts::Game<typename T::Game>;
  // requires search::concepts::Node<typename T::Node, T>;
};

}  // namespace concepts
}  // namespace search
