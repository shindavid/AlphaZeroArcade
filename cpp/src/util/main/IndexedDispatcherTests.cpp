#include "util/GTestUtil.hpp"
#include "util/IndexedDispatcher.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(IndexedDispatcher, BasicDispatch) {
  // Dispatch runtime index to compile-time integral_constant
  std::string result;
  for (int i = 0; i < 4; ++i) {
    util::IndexedDispatcher<4>::call(i, [&](auto ic) {
      constexpr int val = decltype(ic)::value;
      result += std::to_string(val) + " ";
    });
  }
  EXPECT_EQ(result, "0 1 2 3 ");
}

TEST(IndexedDispatcher, ReturnValue) {
  // Test that the dispatch returns a value
  auto square = [](auto ic) -> int {
    constexpr int v = decltype(ic)::value;
    return v * v;
  };

  EXPECT_EQ(util::IndexedDispatcher<5>::call(0, square), 0);
  EXPECT_EQ(util::IndexedDispatcher<5>::call(1, square), 1);
  EXPECT_EQ(util::IndexedDispatcher<5>::call(2, square), 4);
  EXPECT_EQ(util::IndexedDispatcher<5>::call(3, square), 9);
  EXPECT_EQ(util::IndexedDispatcher<5>::call(4, square), 16);
}

TEST(IndexedDispatcher, CompileTimeValue) {
  // Verify the dispatched value is truly compile-time via constexpr arrays
  auto get_fib = [](auto ic) -> int {
    constexpr int v = decltype(ic)::value;
    constexpr int fibs[] = {0, 1, 1, 2, 3, 5, 8, 13};
    return fibs[v];
  };

  EXPECT_EQ(util::IndexedDispatcher<8>::call(0, get_fib), 0);
  EXPECT_EQ(util::IndexedDispatcher<8>::call(5, get_fib), 5);
  EXPECT_EQ(util::IndexedDispatcher<8>::call(7, get_fib), 13);
}

TEST(IndexedDispatcher, SingleElement) {
  int called_with = -1;
  util::IndexedDispatcher<1>::call(0, [&](auto ic) { called_with = decltype(ic)::value; });
  EXPECT_EQ(called_with, 0);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
