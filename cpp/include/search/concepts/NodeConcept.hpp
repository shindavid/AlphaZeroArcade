#pragma once

#include "core/NodeBase.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class N, class Game>
concept Node = requires { requires std::derived_from<N, core::NodeBase<Game>>; };

}  // namespace concepts
}  // namespace search
