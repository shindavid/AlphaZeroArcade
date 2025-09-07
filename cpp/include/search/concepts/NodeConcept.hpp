#pragma once

#include "core/NodeBase.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class N, class EvalSpec>
concept Node = requires { requires std::derived_from<N, core::NodeBase<EvalSpec>>; };

}  // namespace concepts
}  // namespace search
