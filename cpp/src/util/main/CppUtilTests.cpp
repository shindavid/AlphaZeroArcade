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

// --- array_size ---

TEST(CppUtil, ArraySize) {
  std::array<int, 3> a3 = {1, 2, 3};
  std::array<float, 7> a7 = {};
  std::array<double, 1> a1 = {};

  EXPECT_EQ(util::array_size(a3), 3u);
  EXPECT_EQ(util::array_size(a7), 7u);
  EXPECT_EQ(util::array_size(a1), 1u);

  static_assert(util::array_size(std::array<int, 5>{}) == 5);
}

// --- to_std_array ---

TEST(CppUtil, ToStdArray) {
  // scalars only
  auto a = util::to_std_array<int>(1, 2, 3);
  static_assert(std::is_same_v<decltype(a), std::array<int, 3>>);
  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[1], 2);
  EXPECT_EQ(a[2], 3);

  // mix scalar + array
  auto b = util::to_std_array<int>(std::array<int, 2>{4, 5}, 6);
  static_assert(std::is_same_v<decltype(b), std::array<int, 3>>);
  EXPECT_EQ(b[0], 4);
  EXPECT_EQ(b[1], 5);
  EXPECT_EQ(b[2], 6);

  // arrays only
  auto c = util::to_std_array<int>(std::array<int, 2>{7, 8}, std::array<int, 1>{9});
  static_assert(std::is_same_v<decltype(c), std::array<int, 3>>);
  EXPECT_EQ(c[0], 7);
  EXPECT_EQ(c[1], 8);
  EXPECT_EQ(c[2], 9);
}

// --- tuple_hash ---

TEST(CppUtil, TupleHash) {
  // Same tuple => same hash
  auto t1 = std::make_tuple(1, 2, 3);
  auto t2 = std::make_tuple(1, 2, 3);
  EXPECT_EQ(util::tuple_hash(t1), util::tuple_hash(t2));

  // Different tuples => different hashes (very likely)
  auto t3 = std::make_tuple(1, 2, 4);
  EXPECT_NE(util::tuple_hash(t1), util::tuple_hash(t3));

  // Single-element tuple
  auto t4 = std::make_tuple(42);
  auto t5 = std::make_tuple(42);
  EXPECT_EQ(util::tuple_hash(t4), util::tuple_hash(t5));

  // std::hash specialization works too
  using TupleHash = std::hash<std::tuple<int, int, int>>;
  EXPECT_EQ(TupleHash{}(t1), TupleHash{}(t2));
}

// --- dummy_mutex ---

TEST(CppUtil, DummyMutex) {
  util::dummy_mutex m;
  // All operations should be no-ops and not throw/crash
  m.lock();
  m.unlock();
  EXPECT_TRUE(m.try_lock());
  m.unlock();

  // Works with lock_guard (standard interface compliance)
  {
    std::lock_guard<util::dummy_mutex> guard(m);
    // nothing to assert, just veryfying it compiles and runs
  }
}

// --- PODHash ---

TEST(CppUtil, PODHash) {
  util::PODHash<int> hasher;

  // Same value => same hash
  EXPECT_EQ(hasher(42), hasher(42));
  EXPECT_EQ(hasher(0), hasher(0));

  // Different values => likely different hashes
  EXPECT_NE(hasher(1), hasher(2));

  // Works for larger POD types
  struct Point { int x, y; };
  util::PODHash<Point> point_hasher;
  Point p1{3, 4};
  Point p2{3, 4};
  Point p3{5, 6};
  EXPECT_EQ(point_hasher(p1), point_hasher(p2));
  EXPECT_NE(point_hasher(p1), point_hasher(p3));
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
