#pragma once

#include <array>
#include <cstddef>

namespace util {

template <typename... Ts>
class TypedUnion {
 public:
  TypedUnion() = default;

  template<typename T>
  TypedUnion(const T& value, int type_index) {
    reinterpret_cast<T&>(union_data_) = value;
    type_index_ = type_index;

    // TODO: static_assert that Ts[type_index] is T
  }

 private:
  static constexpr int kDataSize = std::max({sizeof(Ts)...});
  using union_data_t = std::array<std::byte, kDataSize>;

  alignas(Ts...) union_data_t union_data_;
  int type_index_;
};

}  // namespace util
