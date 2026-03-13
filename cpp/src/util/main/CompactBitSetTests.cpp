#include "util/CompactBitSet.hpp"
#include "util/GTestUtil.hpp"
#include "util/Random.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

namespace {

template <class Range>
std::vector<int> collect(Range&& r) {
  std::vector<int> v;
  for (auto i : r) v.push_back(static_cast<int>(i));
  return v;
}

static void ExpectRoughlyUniform(const std::vector<int>& counts, int total_trials,
                                 double tol_frac) {
  ASSERT_GT(total_trials, 0);
  ASSERT_FALSE(counts.empty());
  const double expected = static_cast<double>(total_trials) / counts.size();
  for (size_t i = 0; i < counts.size(); ++i) {
    double c = static_cast<double>(counts[i]);
    double lo = expected * (1.0 - tol_frac);
    double hi = expected * (1.0 + tol_frac);
    EXPECT_LE(lo, c) << "bucket " << i << " too low; got " << c << " expected ~ " << expected;
    EXPECT_GE(hi, c) << "bucket " << i << " too high; got " << c << " expected ~ " << expected;
  }
}

}  // namespace

// --------------------- One-word cases (N <= 64) ---------------------

TEST(CompactBitSet, SizeAndDefaultState_OneWord) {
  using BS = util::CompactBitSet<10>;
  BS bs;
  EXPECT_EQ(BS::size(), 10u);
  EXPECT_TRUE(bs.none());
  EXPECT_FALSE(bs.any());
  EXPECT_FALSE(bs.all());
  EXPECT_EQ(bs.count(), 0u);
}

TEST(CompactBitSet, SetResetAndTest_OneWord) {
  using BS = util::CompactBitSet<12>;
  BS bs;
  bs.set(0).set(5).set(11);
  EXPECT_TRUE(bs.test(0));
  EXPECT_TRUE(bs[5]);
  EXPECT_TRUE(bs.test(11));
  EXPECT_EQ(bs.count(), 3u);

  bs.reset(5);
  EXPECT_FALSE(bs.test(5));
  EXPECT_EQ(bs.count(), 2u);

  bs.set();
  EXPECT_TRUE(bs.all());
  EXPECT_EQ(bs.count(), BS::size());
  bs.reset();
  EXPECT_TRUE(bs.none());
}

TEST(CompactBitSet, BitwiseOps_OneWord) {
  using BS = util::CompactBitSet<16>;
  BS a, b;

  a.set(0).set(5).set(12);
  b.set(0).set(4).set(12);

  BS c_and = a & b;
  BS c_or = a | b;
  BS c_xor = a ^ b;

  EXPECT_TRUE(c_and.test(0));
  EXPECT_TRUE(c_and.test(12));
  EXPECT_FALSE(c_and.test(4));
  EXPECT_FALSE(c_and.test(5));

  EXPECT_TRUE(c_or.test(4));
  EXPECT_TRUE(c_or.test(5));
  EXPECT_TRUE(c_or.test(12));
  EXPECT_TRUE(c_or.test(0));

  EXPECT_TRUE(c_xor.test(4));
  EXPECT_TRUE(c_xor.test(5));
  EXPECT_FALSE(c_xor.test(0));
  EXPECT_FALSE(c_xor.test(12));

  BS zero;
  BS all = ~zero;
  EXPECT_TRUE(all.all());
  EXPECT_EQ(all.count(), BS::size());
}

TEST(CompactBitSet, IterationOnOff_OneWord) {
  using BS = util::CompactBitSet<10>;
  BS bs;
  bs.set(0).set(1).set(2).set(4).set(7);

  auto on = collect(bs.on_indices());
  auto off = collect(bs.off_indices());

  std::vector<int> want_on = {0, 1, 2, 4, 7};
  std::vector<int> want_off = {3, 5, 6, 8, 9};

  EXPECT_EQ(on, want_on);
  EXPECT_EQ(off, want_off);
}

TEST(CompactBitSet, RankSelectAndPrefixCount_OneWord) {
  using BS = util::CompactBitSet<20>;
  BS bs;
  bs.set(0).set(2).set(3).set(10).set(15);

  EXPECT_EQ(bs.get_nth_on_index(0), 0);
  EXPECT_EQ(bs.get_nth_on_index(1), 2);
  EXPECT_EQ(bs.get_nth_on_index(2), 3);
  EXPECT_EQ(bs.get_nth_on_index(3), 10);
  EXPECT_EQ(bs.get_nth_on_index(4), 15);

  EXPECT_EQ(bs.count_on_indices_before(0), 0);
  EXPECT_EQ(bs.count_on_indices_before(1), 1);
  EXPECT_EQ(bs.count_on_indices_before(2), 1);
  EXPECT_EQ(bs.count_on_indices_before(3), 2);
  EXPECT_EQ(bs.count_on_indices_before(4), 3);
  EXPECT_EQ(bs.count_on_indices_before(11), 4);
  EXPECT_EQ(bs.count_on_indices_before(20), 5);
}

TEST(CompactBitSet, ToStringNatural_OneWord) {
  using BS = util::CompactBitSet<8>;
  BS bs;
  bs.set(0).set(3).set(7);
  EXPECT_EQ(bs.to_string_natural(), std::string("10010001"));
}

// --------------------- Multi-word cases (N > 64) ---------------------

TEST(CompactBitSet, SizeAndDefaultState_MultiWord) {
  using BS = util::CompactBitSet<70>;
  BS bs;
  EXPECT_EQ(BS::size(), 70u);
  EXPECT_TRUE(bs.none());
  EXPECT_EQ(bs.count(), 0u);
}

TEST(CompactBitSet, SetResetAndTest_MultiWord) {
  using BS = util::CompactBitSet<70>;
  BS bs;
  bs.set(0).set(1).set(2).set(64).set(65).set(69);
  EXPECT_TRUE(bs.test(0));
  EXPECT_TRUE(bs.test(64));
  EXPECT_TRUE(bs.test(69));
  EXPECT_EQ(bs.count(), 6u);

  bs.reset(65);
  EXPECT_FALSE(bs.test(65));
  EXPECT_EQ(bs.count(), 5u);
}

TEST(CompactBitSet, BitwiseOpsAndTailMask_MultiWord) {
  using BS = util::CompactBitSet<70>;
  BS zero;
  BS all = ~zero;
  EXPECT_TRUE(all.any());
  EXPECT_TRUE(all.all());
  EXPECT_EQ(all.count(), 70u);

  BS bs;
  bs.set(0).set(64).set(69);
  BS mask;
  mask.set();
  BS anded = bs & mask;
  EXPECT_EQ(anded.count(), 3u);
  EXPECT_TRUE(anded.test(0));
  EXPECT_TRUE(anded.test(64));
  EXPECT_TRUE(anded.test(69));

  BS x, y;
  x.set(69);
  y.set(64);
  x |= y;
  EXPECT_TRUE(x.test(64));
  EXPECT_TRUE(x.test(69));
  x ^= y;
  EXPECT_FALSE(x.test(64));
  EXPECT_TRUE(x.test(69));
}

TEST(CompactBitSet, IterationOnOff_MultiWord) {
  using BS = util::CompactBitSet<70>;
  BS bs;
  bs.set(0).set(3).set(4).set(64).set(66).set(69);

  auto on = collect(bs.on_indices());
  auto off = collect(bs.off_indices());

  std::vector<int> want_on = {0, 3, 4, 64, 66, 69};
  EXPECT_EQ(on, want_on);

  EXPECT_TRUE(std::binary_search(off.begin(), off.end(), 1));
  EXPECT_TRUE(std::binary_search(off.begin(), off.end(), 2));
  EXPECT_TRUE(std::binary_search(off.begin(), off.end(), 65));
  EXPECT_TRUE(std::binary_search(off.begin(), off.end(), 68));
  EXPECT_TRUE(std::binary_search(off.begin(), off.end(), 67));
  EXPECT_TRUE(std::binary_search(off.begin(), off.end(), 63));
  EXPECT_TRUE(std::binary_search(off.begin(), off.end(), 62));
}

TEST(CompactBitSet, RankSelectAndPrefixCount_MultiWord) {
  using BS = util::CompactBitSet<70>;
  BS bs;
  bs.set(0).set(2).set(3).set(10).set(64).set(66).set(69);

  EXPECT_EQ(bs.get_nth_on_index(0), 0);
  EXPECT_EQ(bs.get_nth_on_index(1), 2);
  EXPECT_EQ(bs.get_nth_on_index(2), 3);
  EXPECT_EQ(bs.get_nth_on_index(3), 10);
  EXPECT_EQ(bs.get_nth_on_index(4), 64);
  EXPECT_EQ(bs.get_nth_on_index(5), 66);
  EXPECT_EQ(bs.get_nth_on_index(6), 69);

  EXPECT_EQ(bs.count_on_indices_before(0), 0);
  EXPECT_EQ(bs.count_on_indices_before(1), 1);
  EXPECT_EQ(bs.count_on_indices_before(4), 3);
  EXPECT_EQ(bs.count_on_indices_before(11), 4);
  EXPECT_EQ(bs.count_on_indices_before(64), 4);
  EXPECT_EQ(bs.count_on_indices_before(65), 5);
  EXPECT_EQ(bs.count_on_indices_before(70), 7);
}

TEST(CompactBitSet, EqualityAcrossWords) {
  using BS = util::CompactBitSet<70>;
  BS a, b;
  a.set(0).set(64).set(69);
  b.set(69).set(0).set(64);
  EXPECT_TRUE(a == b);
  b.reset(64);
  EXPECT_FALSE(a == b);
}

// --------------------- choose_random_on_index ---------------------

TEST(CompactBitSet_Random, ChooseRandomOnIndex_OneWord_BasicAndUniform) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<20>;
  BS bs;
  std::vector<int> on = {0, 3, 7, 12, 15};
  for (int k : on) bs.set(k);

  const int trials = 20000;
  std::vector<int> hist(on.size(), 0);

  for (int t = 0; t < trials; ++t) {
    int idx = bs.choose_random_on_index();
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, static_cast<int>(BS::size()));
    ASSERT_TRUE(bs.test(static_cast<size_t>(idx)));

    auto it = std::find(on.begin(), on.end(), idx);
    ASSERT_NE(it, on.end());
    ++hist[static_cast<int>(std::distance(on.begin(), it))];
  }

  ExpectRoughlyUniform(hist, trials, 0.12);
}

TEST(CompactBitSet_Random, ChooseRandomOnIndex_MultiWord_BasicAndUniform) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<70>;
  BS bs;
  std::vector<int> on = {0, 2, 3, 10, 64, 66, 69};
  for (int k : on) bs.set(k);

  const int trials = 28000;
  std::vector<int> hist(on.size(), 0);

  for (int t = 0; t < trials; ++t) {
    int idx = bs.choose_random_on_index();
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, static_cast<int>(BS::size()));
    ASSERT_TRUE(bs.test(static_cast<size_t>(idx)));

    auto it = std::find(on.begin(), on.end(), idx);
    ASSERT_NE(it, on.end());
    ++hist[static_cast<int>(std::distance(on.begin(), it))];
  }

  ExpectRoughlyUniform(hist, trials, 0.10);
}

// --------------------- randomly_zero_out ---------------------

TEST(CompactBitSet_Random, RandomlyZeroOut_OneWord_CountAndMembership) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<24>;
  BS bs;
  for (int i = 0; i < 16; ++i) bs.set(i);

  BS original = bs;

  bs.randomly_zero_out(5);

  EXPECT_EQ(bs.count(), original.count() - 5);

  int cleared = 0;
  for (int i = 0; i < 16; ++i) {
    if (original.test(i) && !bs.test(i)) ++cleared;
  }
  EXPECT_EQ(cleared, 5);
}

TEST(CompactBitSet_Random, RandomlyZeroOut_OneWord_UniformSingles) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<20>;
  std::vector<int> base_on(10);
  std::iota(base_on.begin(), base_on.end(), 0);

  const int choices = static_cast<int>(base_on.size());
  const int trials = 20000;
  std::vector<int> hist(choices, 0);

  for (int t = 0; t < trials; ++t) {
    BS bs;
    for (int k : base_on) bs.set(k);

    bs.randomly_zero_out(1);

    int cleared = -1;
    for (int k : base_on) {
      if (!bs.test(static_cast<size_t>(k))) {
        cleared = k;
        break;
      }
    }
    ASSERT_NE(cleared, -1);

    ++hist[cleared];
  }

  ExpectRoughlyUniform(hist, trials, 0.12);
}

TEST(CompactBitSet_Random, RandomlyZeroOut_MultiWord_CountAndUniformSingles) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<70>;
  std::vector<int> base_on;
  for (int i = 0; i < 32; ++i) base_on.push_back(i);
  for (int i = 64; i < 70; ++i) base_on.push_back(i);

  const int choices = static_cast<int>(base_on.size());
  const int trials = 38000;
  std::vector<int> hist(choices, 0);

  for (int t = 0; t < trials; ++t) {
    BS bs;
    for (int k : base_on) bs.set(k);
    BS original = bs;

    bs.randomly_zero_out(1);

    EXPECT_EQ(bs.count(), original.count() - 1);

    int cleared_idx = -1;
    for (int i = 0; i < choices; ++i) {
      int k = base_on[i];
      if (original.test(static_cast<size_t>(k)) && !bs.test(static_cast<size_t>(k))) {
        cleared_idx = i;
        break;
      }
    }
    ASSERT_NE(cleared_idx, -1);
    ++hist[cleared_idx];
  }

  ExpectRoughlyUniform(hist, trials, 0.10);
}

TEST(CompactBitSet_Random, RandomlyZeroOut_AllBits) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<33>;
  BS bs;
  for (size_t i = 0; i < BS::size(); ++i) bs.set(i);
  const auto before = bs.count();

  bs.randomly_zero_out(static_cast<int>(before));
  EXPECT_TRUE(bs.none());
  EXPECT_EQ(bs.count(), 0u);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
