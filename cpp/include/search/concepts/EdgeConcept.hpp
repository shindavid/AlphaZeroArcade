#pragma once

#include "search/EdgeBase.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class E>
concept Edge = requires { requires std::derived_from<E, search::EdgeBase>; };

}  // namespace concepts
}  // namespace search
