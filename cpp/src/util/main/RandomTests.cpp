#include "util/GTestUtil.hpp"
#include "util/Random.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

namespace {

template <typename T>
void test_zero_out() {
  util::Random::set_seed(1);
  std::array<int, 10> counts = {};
  std::array<T, 10> orig_a = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1};

  constexpr int N = 10000;
  for (int i = 0; i < N; ++i) {
    std::array<T, 10> a = orig_a;
    util::Random::zero_out(a.begin(), a.end(), 4);
    for (size_t j = 0; j < a.size(); ++j) {
      if (a[j]) counts[j]++;
    }
  }
  for (size_t i = 0; i < counts.size(); ++i) {
    if (orig_a[i]) {
      double pct = counts[i] * 1.0 / N;
      double error = std::abs(pct - 0.5);
      EXPECT_LT(error, 0.01);
    } else {
      EXPECT_EQ(counts[i], 0);
    }
  }
}

}  // namespace

TEST(Random, zero_out) {
  test_zero_out<bool>();
  test_zero_out<int>();
}

TEST(Random, uniform_sample_range) {
  util::Random::set_seed(42);

  for (int trial = 0; trial < 1000; ++trial) {
    int v = util::Random::uniform_sample(5, 10);
    EXPECT_GE(v, 5);
    EXPECT_LT(v, 10);
  }
}

TEST(Random, uniform_sample_single_value) {
  util::Random::set_seed(42);
  // [5, 6) should always return 5
  for (int trial = 0; trial < 100; ++trial) {
    EXPECT_EQ(util::Random::uniform_sample(5, 6), 5);
  }
}

TEST(Random, uniform_real_range) {
  util::Random::set_seed(42);

  for (int trial = 0; trial < 1000; ++trial) {
    float v = util::Random::uniform_real(1.0f, 2.0f);
    EXPECT_GE(v, 1.0f);
    EXPECT_LT(v, 2.0f);
  }
}

TEST(Random, shuffle_is_permutation) {
  util::Random::set_seed(42);

  std::vector<int> v(20);
  std::iota(v.begin(), v.end(), 0);
  std::vector<int> original = v;

  util::Random::shuffle(v.begin(), v.end());

  // Same elements, potentially different order
  std::vector<int> sorted_v = v;
  std::sort(sorted_v.begin(), sorted_v.end());
  EXPECT_EQ(sorted_v, original);

  // Very unlikely to still be sorted (20! is huge)
  EXPECT_NE(v, original);
}

TEST(Random, weighted_sample_distribution) {
  util::Random::set_seed(42);

  // Weights: [1, 2, 3, 4] => expected proportions [0.1, 0.2, 0.3, 0.4]
  std::array<float, 4> weights = {1.0f, 2.0f, 3.0f, 4.0f};
  constexpr int N = 40000;
  std::array<int, 4> counts = {};

  for (int i = 0; i < N; ++i) {
    int idx = util::Random::weighted_sample(weights.begin(), weights.end());
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, 4);
    counts[idx]++;
  }

  float total_weight = 10.0f;
  for (int i = 0; i < 4; ++i) {
    double expected = weights[i] / total_weight;
    double actual = counts[i] * 1.0 / N;
    EXPECT_NEAR(actual, expected, 0.02) << "bucket " << i;
  }
}

TEST(Random, chunked_shuffle) {
  util::Random::set_seed(42);

  std::vector<int> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  util::Random::chunked_shuffle(v.begin(), v.end(), 3);

  // Chunks of 3 should be preserved internally
  // Each chunk [0,1,2], [3,4,5], [6,7,8], [9,10,11] should stay together
  // Find where chunk starting with 0 ended up:
  for (size_t i = 0; i < v.size(); i += 3) {
    std::vector<int> chunk(v.begin() + i, v.begin() + i + 3);
    // Chunk should be 3 consecutive integers
    EXPECT_EQ(chunk[1], chunk[0] + 1) << "at i=" << i;
    EXPECT_EQ(chunk[2], chunk[0] + 2) << "at i=" << i;
    // First element should be a multiple of 3
    EXPECT_EQ(chunk[0] % 3, 0) << "at i=" << i;
  }
}

TEST(Random, exponential_positive) {
  util::Random::set_seed(42);

  for (int trial = 0; trial < 100; ++trial) {
    float v = util::Random::exponential(1.0f);
    EXPECT_GT(v, 0.0f);
  }
}

TEST(Random, seeded_prng_deterministic) {
  util::Random::set_seed(123);
  std::vector<int> seq1;
  for (int i = 0; i < 20; ++i) {
    seq1.push_back(util::Random::uniform_sample(0, 1000));
  }

  util::Random::set_seed(123);
  std::vector<int> seq2;
  for (int i = 0; i < 20; ++i) {
    seq2.push_back(util::Random::uniform_sample(0, 1000));
  }

  EXPECT_EQ(seq1, seq2);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
