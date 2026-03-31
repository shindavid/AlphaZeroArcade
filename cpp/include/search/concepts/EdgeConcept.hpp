#pragma once

#include "search/EdgeBase.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class E, class EvalSpec>
concept Edge = requires { requires std::derived_from<E, search::EdgeBase<EvalSpec>>; };

}  // namespace concepts
}  // namespace search
