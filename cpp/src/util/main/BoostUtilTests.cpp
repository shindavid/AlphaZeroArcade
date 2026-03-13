#include "util/BoostUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/Random.hpp"

#include <gtest/gtest.h>

#include <map>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

TEST(BoostUtil, get_random_set_index) {
  util::Random::set_seed(1);

  int n = 10;
  boost::dynamic_bitset<> bitset(n);
  int index;

  index = boost_util::get_random_set_index(bitset);
  EXPECT_EQ(index, -1);

  bitset.set(1);
  bitset.set(3);
  bitset.set(5);
  bitset.set(7);
  bitset.set(9);

  int N = 1000;
  using count_map_t = std::map<int, int>;
  count_map_t counts;

  for (int i = 0; i < N; ++i) {
    index = boost_util::get_random_set_index(bitset);
    counts[index]++;
    EXPECT_LT(index, n);
    EXPECT_GE(index, 0);
    EXPECT_TRUE(bitset.test(index));
  }

  int count = bitset.count();
  int c = bitset.find_first();
  int i = count;
  while (i-- > 0) {
    EXPECT_GT(counts[c], N / (2 * count));
    c = bitset.find_next(c);
  }
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
