#pragma once

#include <type_traits>

namespace util {

/*
 * This identity function is useful for declaring required members in concepts.
 */
template <class T> std::decay_t<T> decay_copy(T&&);

}  // namespace util
