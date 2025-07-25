#pragma once

#include <cstdint>
#include <tuple>
#include <utility>

/*
 * Adapted from: https://codereview.stackexchange.com/q/269320
 */
namespace mp {

template <typename...>
struct TypeList;

template <>
struct TypeList<> {};

template <typename Head, typename... Tails>
struct TypeList<Head, Tails...> {
  using head = Head;
  using tails = TypeList<Tails...>;
};

template <typename Base, typename T>
struct AllDerivedFrom;

template <typename Base>
struct AllDerivedFrom<Base, TypeList<>> {
  static constexpr bool value = true;
};

template <typename Base, typename Head, typename... Tails>
struct AllDerivedFrom<Base, TypeList<Head, Tails...>> {
  static constexpr bool value =
    std::is_base_of_v<Base, Head> && AllDerivedFrom<Base, TypeList<Tails...>>::value;
};

template <typename T>
concept IsTypeList = requires(T t) { []<typename... Ts>(TypeList<Ts...>&) {}(t); };

template <typename T, typename Base>
concept IsTypeListOf = IsTypeList<T> && AllDerivedFrom<Base, T>::value;

template <template <typename> typename Pred, typename T>
struct AllSatisfyConcept;

template <template <typename> typename Pred>
struct AllSatisfyConcept<Pred, TypeList<>> : std::bool_constant<true> {};

template <template <typename> typename Pred, typename Head, typename... Tails>
struct AllSatisfyConcept<Pred, TypeList<Head, Tails...>>
    : std::bool_constant<Pred<Head>::value && AllSatisfyConcept<Pred, TypeList<Tails...>>::value> {
};

/*
 * NOTE(dshin): VSCode seems to get confused by the IsTypeList<T> requirement here, even though
 * gcc compiles it fine. I'm commenting it out to make the IDE happy.
 */
template <typename T, template <typename> typename Pred>
concept IsTypeListSatisfying = /*IsTypeList<T> &&*/ AllSatisfyConcept<Pred, T>::value;

// length of a typelist

template <typename TList>
struct Length;

template <typename... Types>
struct Length<TypeList<Types...>> {
  static constexpr std::size_t value = sizeof...(Types);
};

template <typename TList>
inline constexpr std::size_t Length_v = Length<TList>::value;

// indexed access

template <typename TList, std::size_t index>
struct TypeAt;

template <typename Head, typename... Tails>
struct TypeAt<TypeList<Head, Tails...>, 0> {
  using type = Head;
};

template <typename Head, typename... Tails, std::size_t index>
struct TypeAt<TypeList<Head, Tails...>, index> {
  static_assert(index < sizeof...(Tails) + 1, "index out of range");
  using type = TypeAt<TypeList<Tails...>, index - 1>::type;
};

template <typename TList, std::size_t index>
using TypeAt_t = TypeAt<TList, index>::type;

// indexof

template <typename TList, typename T>
struct IndexOf;

template <typename T>
struct IndexOf<TypeList<>, T> {
  static constexpr std::size_t value = -1;
};

template <typename TList, typename T>
inline constexpr std::size_t IndexOf_v = IndexOf<TList, T>::value;

template <typename... Tails, typename T>
struct IndexOf<TypeList<T, Tails...>, T> {
  static constexpr std::size_t value = 0;
};

template <typename Head, typename... Tails, typename T>
struct IndexOf<TypeList<Head, Tails...>, T> {
  static constexpr std::size_t value =
    std::is_same_v<Head, T>
      ? 0
      : (IndexOf_v<TypeList<Tails...>, T> == -1 ? -1 : IndexOf_v<TypeList<Tails...>, T> + 1);
};

// apply

template <typename TList, template <typename> typename F>
struct Apply;

template <template <typename> typename F>
struct Apply<TypeList<>, F> {
  using type = TypeList<>;
};

template <template <typename> typename F, typename Head, typename... Tails>
struct Apply<TypeList<Head, Tails...>, F> {
  using type = TypeList<F<Head>, typename Apply<TypeList<Tails...>, F>::type>;
};

template <typename TList, template <typename> typename F>
using Apply_t = Apply<TList, F>::type;

// maxsizeof

template <typename TList>
struct MaxSizeOf;

template <typename T>
struct MaxSizeOf<TypeList<T>> {
  static constexpr std::size_t value = sizeof(T);
};

template <typename Head, typename... Tails>
struct MaxSizeOf<TypeList<Head, Tails...>> {
  static constexpr std::size_t value = sizeof(Head) > MaxSizeOf<TypeList<Tails...>>::value
                                         ? sizeof(Head)
                                         : MaxSizeOf<TypeList<Tails...>>::value;
};

template <typename TList>
inline constexpr std::size_t MaxSizeOf_v = MaxSizeOf<TList>::value;

template <typename T>
struct MaxOf {};

template <typename T, T I>
struct MaxOf<std::integer_sequence<T, I>> {
  static constexpr T value = I;
};

template <typename T, T I, T... Is>
struct MaxOf<std::integer_sequence<T, I, Is...>> {
  static constexpr T value = std::max(I, MaxOf<std::integer_sequence<T, Is...>>::value);
};

template <typename T>
constexpr auto MaxOf_v = MaxOf<T>::value;

// static for loop
//
// Usage:
//
// template<int i> func() {}
//
// constexpr_for<0, 10, 1>([](auto i) { func<i>(); }
//
// The above is equivalent to the following (which is not c++ legal):
//
// for (int i = 0; i < 10; i += 1) { func<i>(); }
//
// See: https://artificial-mind.net/blog/2020/10/31/constexpr-for

template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}
}  // namespace mp
