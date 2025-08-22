#pragma once

#include "search/NodeBase.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class N, class Traits>
concept Node = requires { requires std::derived_from<N, search::NodeBase<Traits>>; };

}  // namespace concepts
}  // namespace search
