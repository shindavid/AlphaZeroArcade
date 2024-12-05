#pragma once

#include <util/Asserts.hpp>
#include <util/MetaProgramming.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace util {

/*
 * TypedUnion<Ts...> is similar to std::variant<Ts...>
 *
 * It stores the value of one of the types in Ts..., along with a runtime int index indicating which
 * type is being stored.
 */
template <typename... Ts>
class TypedUnion {
 public:
  using TypeList = mp::TypeList<Ts...>;
  template<int I> using TypeAt = mp::TypeAt_t<TypeList, I>;

  static_assert((std::is_trivially_destructible_v<Ts> && ...),
                "All types in Ts... must have trivial destructors.");

  TypedUnion() = default;

  template<typename T>
  TypedUnion(const T& value, int type_index=0) {
    new (&union_data_) T(value);
    type_index_ = type_index;

    // TODO: static_assert that Ts[type_index] is T
  }

  int type_index() const { return type_index_; }

  // asserts that this->type_index() == I, and then returns a reference to the stored value of type
  template <int I>
  auto& get() {
    static_assert(0 <= I && I < sizeof...(Ts), "Index out of range");
    util::release_assert(type_index_ == I, "Unexpected type-index: %d != %d", type_index_, I);
    return *reinterpret_cast<TypeAt<I>*>(&union_data_);
  }

  // asserts that this->type_index() == I, and then returns a reference to the stored value of type
  template <int I>
  const auto& get() const {
    static_assert(0 <= I && I < sizeof...(Ts), "Index out of range");
    util::release_assert(type_index_ == I, "Unexpected type-index: %d != %d", type_index_, I);
    return *reinterpret_cast<const TypeAt<I>*>(&union_data_);
  }

  /*
   * This object stores an instance of one particular type in Ts..., say, T.
   *
   * This method performs the function call f(t), where t is the stored instance of T.
   *
   * Example usage:
   *
   *  TypedUnion<int, double, std::string> u(42, 0);
   *  u.call([](auto& value) { std::cout << value << std::endl; }
   */
  template <typename F>
  auto call(F&& f) {
    using R = decltype(f(std::declval<TypeAt<0>&>()));

    // TODO: static-assert that R is the same for all Ts...

    if constexpr (std::is_same_v<R, void>) {
      call_impl(std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>{});
    } else {
      std::optional<R> result;
      call_impl(std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>{}, result);
      return *result;
    }
  }

  template <typename F>
  auto call(F&& f) const {
    using R = decltype(f(std::declval<TypeAt<0>&>()));

    // TODO: static-assert that R is the same for all Ts...

    if constexpr (std::is_same_v<R, void>) {
      call_impl(std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>{});
    } else {
      std::optional<R> result;
      call_impl(std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>{}, result);
      return *result;
    }
  }

 private:
  static constexpr int kDataSize = mp::MaxSizeOf_v<TypeList>;
  using union_data_t = std::array<std::byte, kDataSize>;

  template <typename F, std::size_t... Is, typename R>
  void call_impl(F&& f, std::index_sequence<Is...>, std::optional<R>& result) {
    bool handled = false;
    // Fold expression to iterate over types
    (void)std::initializer_list<int>{
        (type_index_ == Is
            ? (handled = true, result = f(*reinterpret_cast<TypeAt<Is>*>(&union_data_)), 0)
            : 0)...};
    util::release_assert(handled, "Invalid type_index_: %d", type_index_);
  }

  template <typename F, std::size_t... Is>
  void call_impl(F&& f, std::index_sequence<Is...>) {
    bool handled = false;
    // Fold expression to iterate over types
    (void)std::initializer_list<int>{
        (type_index_ == Is
            ? (handled = true, f(*reinterpret_cast<TypeAt<Is>*>(&union_data_)), 0)
            : 0)...};
    util::release_assert(handled, "Invalid type_index_: %d", type_index_);
  }

  template <typename F, std::size_t... Is, typename R>
  void call_impl(F&& f, std::index_sequence<Is...>, std::optional<R>& result) const {
    bool handled = false;
    // Fold expression to iterate over types
    (void)std::initializer_list<int>{
        (type_index_ == Is
            ? (handled = true, result = f(*reinterpret_cast<const TypeAt<Is>*>(&union_data_)), 0)
            : 0)...};
    util::release_assert(handled, "Invalid type_index_: %d", type_index_);
  }

  template <typename F, std::size_t... Is>
  void call_impl(F&& f, std::index_sequence<Is...>) const {
    bool handled = false;
    // Fold expression to iterate over types
    (void)std::initializer_list<int>{
        (type_index_ == Is
            ? (handled = true, f(*reinterpret_cast<const TypeAt<Is>*>(&union_data_)), 0)
            : 0)...};
    util::release_assert(handled, "Invalid type_index_: %d", type_index_);
  }

  alignas(Ts...) union_data_t union_data_;
  int type_index_;
};

// TODO: Specialization of TypedUnion for the case where there is one type. In that case, we don't
// need type_index_.

}  // namespace util
