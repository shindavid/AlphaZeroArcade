#pragma once

#include "search/EdgeBase.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class E, class Spec>
concept Edge = requires { requires std::derived_from<E, search::EdgeBase<Spec>>; };

}  // namespace concepts
}  // namespace search
