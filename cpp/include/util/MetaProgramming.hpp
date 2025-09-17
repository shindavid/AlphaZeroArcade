#pragma once

#include <algorithm>
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

// If F is an alias template: template<class T> using F = /*mapped type*/;
template <template <typename> typename F, typename... Ts>
struct Apply<TypeList<Ts...>, F> {
  using type = TypeList<typename F<Ts>::type...>;
};

template <typename TList, template <typename> typename F>
using Apply_t = typename Apply<TList, F>::type;

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

// concat

template <typename... TLists>
struct Concat;

template <>
struct Concat<> {
  using type = TypeList<>;
};

template <typename... Types>
struct Concat<TypeList<Types...>> {
  using type = TypeList<Types...>;
};

template <typename... HeadTypes, typename... TailTypes, typename... TLists>
struct Concat<TypeList<HeadTypes...>, TypeList<TailTypes...>, TLists...> {
  using type = typename Concat<TypeList<HeadTypes..., TailTypes...>, TLists...>::type;
};

template <typename... TLists>
using Concat_t = typename Concat<TLists...>::type;

// rebind

template <class TList, template <class...> class NewTemplate>
struct Rebind;

template <template <class...> class OldTemplate, class... T, template <class...> class NewTemplate>
struct Rebind<OldTemplate<T...>, NewTemplate> {
  using type = NewTemplate<T...>;
};

// Convenience alias for Rebind
template <class TList, template <class...> class NewTemplate>
using Rebind_t = Rebind<TList, NewTemplate>::type;

// filter

template <typename TList, template <typename> typename Pred>
struct Filter;

template <template <typename> typename Pred>
struct Filter<TypeList<>, Pred> {
  using type = TypeList<>;
};

template <template <typename> typename Pred, typename Head, typename... Tails>
struct Filter<TypeList<Head, Tails...>, Pred> {
 private:
  using TailResult = typename Filter<TypeList<Tails...>, Pred>::type;

 public:
  using type = std::conditional_t<Pred<Head>::value, TypeList<Head, TailResult>, TailResult>;
};

template <typename TList, template <typename> typename Pred>
using Filter_t = typename Filter<TList, Pred>::type;

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

// for_each
//
// Usage:
//
// using TL = TypeList<int, float, double>;
// for_each<TL>([]<class T> {
//   do something with T
// });

template <class T, class F>
constexpr void invoke_for_type(F&& f) {
  if constexpr (requires { std::forward<F>(f).template operator()<T>(); }) {
    std::forward<F>(f).template operator()<T>();
  } else {
    std::forward<F>(f)(std::type_identity<T>{});
  }
}

// for_each<TypeList<...>>(f)
template <class TL, class F>
constexpr void for_each(F&& f) {
  [&]<class... Ts>(TypeList<Ts...>) { (invoke_for_type<Ts>(std::forward<F>(f)), ...); }(TL{});
}

}  // namespace mp
