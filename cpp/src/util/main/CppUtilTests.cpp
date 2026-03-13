#include "util/CppUtil.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

// --- StringLiteral tests ---

TEST(CppUtil, StringLiteralEquality) {
  static_assert(util::str_equal<"foo", "foo">());
  static_assert(!util::str_equal<"foo", "bar">());
  static_assert(!util::str_equal<"foo", "fooo">());
}

// --- StringLiteralSequence contains ---

TEST(CppUtil, StringLiteralSequenceContains) {
  using Seq = util::StringLiteralSequence<"alpha", "beta", "gamma">;
  static_assert(util::string_literal_sequence_contains_v<Seq, "alpha">);
  static_assert(util::string_literal_sequence_contains_v<Seq, "beta">);
  static_assert(util::string_literal_sequence_contains_v<Seq, "gamma">);
  static_assert(!util::string_literal_sequence_contains_v<Seq, "delta">);
  SUCCEED();
}

// --- int_sequence contains ---

TEST(CppUtil, IntSequenceContains) {
  using Seq = util::int_sequence<1, 3, 5>;
  static_assert(util::int_sequence_contains_v<Seq, 1>);
  static_assert(util::int_sequence_contains_v<Seq, 3>);
  static_assert(util::int_sequence_contains_v<Seq, 5>);
  static_assert(!util::int_sequence_contains_v<Seq, 2>);
  static_assert(!util::int_sequence_contains_v<Seq, 0>);
  SUCCEED();
}

// --- no_overlap ---

TEST(CppUtil, NoOverlapStringLiteralSequence) {
  using S1 = util::StringLiteralSequence<"foo", "bar">;
  using S2 = util::StringLiteralSequence<"baz", "qux">;
  using S3 = util::StringLiteralSequence<"bar", "quux">;

  static_assert(util::no_overlap_v<S1, S2>);
  static_assert(!util::no_overlap_v<S1, S3>);
  SUCCEED();
}

TEST(CppUtil, NoOverlapIntSequence) {
  using S1 = util::int_sequence<1, 2, 3>;
  using S2 = util::int_sequence<4, 5, 6>;
  using S3 = util::int_sequence<3, 7>;

  static_assert(util::no_overlap_v<S1, S2>);
  static_assert(!util::no_overlap_v<S1, S3>);
  SUCCEED();
}

// --- concat sequences ---

TEST(CppUtil, ConcatIntSequence) {
  using A = util::int_sequence<1, 2>;
  using B = util::int_sequence<3, 4>;
  using C = util::concat_int_sequence_t<A, B>;

  static_assert(std::is_same_v<C, std::integer_sequence<int, 1, 2, 3, 4>>);
  SUCCEED();
}

TEST(CppUtil, ConcatStringLiteralSequence) {
  using A = util::StringLiteralSequence<"foo">;
  using B = util::StringLiteralSequence<"bar">;
  using C = util::concat_string_literal_sequence_t<A, B>;

  static_assert(util::string_literal_sequence_contains_v<C, "foo">);
  static_assert(util::string_literal_sequence_contains_v<C, "bar">);
  SUCCEED();
}

// --- Time conversions ---

TEST(CppUtil, TimeConversions) {
  EXPECT_EQ(util::s_to_ns(1), 1'000'000'000LL);
  EXPECT_EQ(util::ms_to_ns(1), 1'000'000LL);
  EXPECT_EQ(util::us_to_ns(1), 1'000LL);
  EXPECT_EQ(util::s_to_ns(0), 0);

  auto dur = std::chrono::milliseconds(42);
  EXPECT_EQ(util::to_ns(dur), 42'000'000LL);
}

// --- get_typename ---

TEST(CppUtil, GetTypename) {
  std::string name = util::get_typename<int>();
  EXPECT_FALSE(name.empty());

  std::string name2 = util::get_typename(3.14);
  EXPECT_FALSE(name2.empty());
}

// --- ReciprocalTable ---

TEST(CppUtil, ReciprocalTable) {
  // ReciprocalTable<N>::get(i) accepts i>=1, returns 1.0f / i
  constexpr int N = 8;
  for (int i = 1; i <= N; ++i) {
    float expected = 1.0f / i;
    EXPECT_FLOAT_EQ(util::ReciprocalTable<N>::get(i), expected) << "i=" << i;
  }
  // For i > N, falls back to direct division
  EXPECT_FLOAT_EQ(util::ReciprocalTable<N>::get(N + 1), 1.0f / (N + 1));
}

// --- std_array_to_string ---

TEST(CppUtil, StdArrayToString) {
  std::array<int, 3> arr = {1, 2, 3};
  std::string s = util::std_array_to_string(arr, "[", ", ", "]");
  EXPECT_EQ(s, "[1, 2, 3]");
}

// --- array_cast ---

TEST(CppUtil, ArrayCast) {
  std::array<int, 3> ints = {1, 2, 3};
  auto floats = util::array_cast<float>(ints);
  static_assert(std::is_same_v<decltype(floats), std::array<float, 3>>);
  EXPECT_FLOAT_EQ(floats[0], 1.0f);
  EXPECT_FLOAT_EQ(floats[1], 2.0f);
  EXPECT_FLOAT_EQ(floats[2], 3.0f);
}

// --- hash_memory ---

TEST(CppUtil, HashMemory) {
  uint32_t a = 0x12345678;
  uint32_t b = 0x12345678;
  uint32_t c = 0x87654321;

  EXPECT_EQ(util::hash_memory<sizeof(a)>(&a), util::hash_memory<sizeof(b)>(&b));
  EXPECT_NE(util::hash_memory<sizeof(a)>(&a), util::hash_memory<sizeof(c)>(&c));
}

// --- integer_sequence helpers ---

TEST(CppUtil, IntegerSequenceProduct) {
  using Seq = std::integer_sequence<int, 2, 3, 5>;
  static_assert(util::integer_sequence_product_v<Seq> == 30);
  SUCCEED();
}

TEST(CppUtil, GetFromIntegerSequence) {
  using Seq = std::integer_sequence<int, 10, 20, 30>;
  EXPECT_EQ((util::get<int, 10, 20, 30>(0, Seq{})), 10);
  EXPECT_EQ((util::get<int, 10, 20, 30>(1, Seq{})), 20);
  EXPECT_EQ((util::get<int, 10, 20, 30>(2, Seq{})), 30);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
