#include "util/GTestUtil.hpp"
#include "util/MetaProgramming.hpp"

#include <gtest/gtest.h>

#include <string>

// --- Length ---

TEST(MetaProgramming, Length) {
  using L0 = mp::TypeList<>;
  using L3 = mp::TypeList<int, float, double>;

  static_assert(mp::Length_v<L0> == 0);
  static_assert(mp::Length_v<L3> == 3);
  SUCCEED();
}

// --- TypeAt ---

TEST(MetaProgramming, TypeAt) {
  using L = mp::TypeList<int, float, double>;

  static_assert(std::is_same_v<mp::TypeAt_t<L, 0>, int>);
  static_assert(std::is_same_v<mp::TypeAt_t<L, 1>, float>);
  static_assert(std::is_same_v<mp::TypeAt_t<L, 2>, double>);
  SUCCEED();
}

// --- IndexOf ---

TEST(MetaProgramming, IndexOf) {
  using L = mp::TypeList<int, float, double>;

  static_assert(mp::IndexOf_v<L, int> == 0);
  static_assert(mp::IndexOf_v<L, float> == 1);
  static_assert(mp::IndexOf_v<L, double> == 2);
  SUCCEED();
}

// --- MaxSizeOf ---

TEST(MetaProgramming, MaxSizeOf) {
  using L = mp::TypeList<char, int, double>;

  static_assert(mp::MaxSizeOf_v<L> == sizeof(double));
  SUCCEED();
}

// --- AllDerivedFrom ---

struct Base {};
struct Child1 : Base {};
struct Child2 : Base {};
struct NotChild {};

TEST(MetaProgramming, AllDerivedFrom) {
  using Good = mp::TypeList<Child1, Child2>;
  using Bad = mp::TypeList<Child1, NotChild>;

  static_assert(mp::AllDerivedFrom<Base, Good>::value);
  static_assert(!mp::AllDerivedFrom<Base, Bad>::value);
  SUCCEED();
}

// --- Concat ---

TEST(MetaProgramming, Concat) {
  using A = mp::TypeList<int, float>;
  using B = mp::TypeList<double>;
  using C = mp::Concat_t<A, B>;

  static_assert(mp::Length_v<C> == 3);
  static_assert(std::is_same_v<mp::TypeAt_t<C, 0>, int>);
  static_assert(std::is_same_v<mp::TypeAt_t<C, 1>, float>);
  static_assert(std::is_same_v<mp::TypeAt_t<C, 2>, double>);
  SUCCEED();
}

// --- Apply ---

template <typename T>
struct AddPointer {
  using type = T*;
};

TEST(MetaProgramming, Apply) {
  using L = mp::TypeList<int, float>;
  using R = mp::Apply_t<L, AddPointer>;

  static_assert(std::is_same_v<mp::TypeAt_t<R, 0>, int*>);
  static_assert(std::is_same_v<mp::TypeAt_t<R, 1>, float*>);
  SUCCEED();
}

// --- Filter ---

template <typename T>
struct IsIntegral : std::is_integral<T> {};

TEST(MetaProgramming, Filter) {
  // Filter uses cons-list nesting: TypeList<Head, TypeList<...>>
  using L = mp::TypeList<int, float, char, double>;
  using F = mp::Filter_t<L, IsIntegral>;

  static_assert(mp::Length_v<F> == 2);
  static_assert(std::is_same_v<mp::TypeAt_t<F, 0>, int>);
  SUCCEED();
}

// --- Rebind ---

TEST(MetaProgramming, Rebind) {
  using L = mp::TypeList<int, float>;
  using R = mp::Rebind_t<L, std::tuple>;

  static_assert(std::is_same_v<R, std::tuple<int, float>>);
  SUCCEED();
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
