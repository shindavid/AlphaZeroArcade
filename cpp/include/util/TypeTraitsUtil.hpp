#pragma once

#include <util/EigenUtil.hpp>

#include <type_traits>

namespace util {

// std::is_trivially_copyable<T>::value is false for Eigen types of fixed size. But such types are
// fine to memcpy. We introduce is_memcpy_safe<T> as our own version of
// std::is_trivially_copyable<T> that is true for fixed-size Eigen types.
template <typename T>
struct is_memcpy_safe : public std::is_trivially_copyable<T> {};

template <eigen_util::concepts::FTensor T>
struct is_memcpy_safe<T> : std::true_type {};

template <eigen_util::concepts::FArray T>
struct is_memcpy_safe<T> : std::true_type {};

template <typename T>
inline constexpr bool is_memcpy_safe_v = is_memcpy_safe<T>::value;

}  // namespace util
