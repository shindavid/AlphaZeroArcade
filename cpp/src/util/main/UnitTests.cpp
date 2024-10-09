#include <util/AllocPool.hpp>
#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <array>
#include <iostream>

template <typename T>
void test_zero_out() {
  util::Random::set_seed(1);
  std::array<int, 10> counts = {};
  std::array<T, 10> orig_a = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1};

  constexpr int N = 10000;
  for (int i = 0; i < N; ++i) {
    std::array<T, 10> a = orig_a;
    util::Random::zero_out(a.begin(), a.end(), 4);
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i]) counts[i]++;
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

TEST(Random, zero_out) {
  test_zero_out<bool>();
  test_zero_out<int>();
}

template<typename Pool>
void test_alloc_pool_helper(Pool& pool, int* sizes, int num_sizes) {
  // add 0, 1, 2, ... to the pool, in chunks given by sizes
  int x = 0;
  for (int i = 0; i < num_sizes; ++i) {
    int size = sizes[i];
    util::pool_index_t idx = pool.alloc(size);
    for (int i = 0; i < size; ++i) {
      pool[idx + i] = x++;
    }
  }

  // validate size
  std::vector<int> vec = pool.to_vector();
  EXPECT_EQ(vec.size(), x);

  // validate contents
  for (int i = 0; i < x; ++i) {
    EXPECT_EQ(vec[i], i);
  }

  // now remove the square elements
  boost::dynamic_bitset<> used_indices(x);
  used_indices.set();
  int y = x;
  for (int i = 0; i * i < x; ++i) {
    used_indices[i*i] = false;
    --y;
  }
  pool.defragment(used_indices);

  // validate size
  vec = pool.to_vector();
  EXPECT_EQ(vec.size(), y);

  // validate contents
  int sqrt = 0;
  int k = 0;
  for (int i = 0; i < x; ++i) {
    if (sqrt * sqrt == i) {
      ++sqrt;
      continue;
    }
    EXPECT_EQ(vec[k], i);
    ++k;
  }
}

TEST(AllocPool, alloc_pool) {
  util::AllocPool<int, 2> pool;

  int sizes1[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  test_alloc_pool_helper(pool, sizes1, sizeof(sizes1) / sizeof(sizes1[0]));
  pool.clear();

  int sizes2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  test_alloc_pool_helper(pool, sizes2, sizeof(sizes2) / sizeof(sizes2[0]));
  pool.clear();

  int sizes3[] = {100};
  test_alloc_pool_helper(pool, sizes3, sizeof(sizes3) / sizeof(sizes3[0]));
  pool.clear();
}

TEST(eigen_util, sort_columns) {
  constexpr int kNumRows = 3;
  constexpr int kNumCols = 5;
  constexpr int kMaxNumCols = 10;
  using Array = Eigen::Array<float, kNumRows, Eigen::Dynamic, 0, kNumRows, kMaxNumCols>;

  Array array{
      {3, 1, 5, 4, 2},
      {30, 10, 50, 40, 20},
      {300, 100, 500, 400, 200},
  };

  array = eigen_util::sort_columns(array);

  int pow10[] = {1, 10, 100};
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      float expected = (c + 1) * pow10[r];
      EXPECT_EQ(array(r, c), expected);
    }
  }
}

TEST(eigen_util, sort_columns_one_element) {
  constexpr int kNumRows = 1;
  constexpr int kNumCols = 1;
  constexpr int kMaxNumCols = 1;
  using Array = Eigen::Array<float, kNumRows, Eigen::Dynamic, 0, kNumRows, kMaxNumCols>;

  Array array{
      {3},
  };

  array = eigen_util::sort_columns(array);

  Array expected{
      {3},
  };

  for (int r = 0; r < array.rows(); r++) {
    for (int c = 0; c < array.cols(); c++) {
      EXPECT_EQ(array(r, c), expected(r, c));
    }
  }
}

TEST(eigen_util, softmax_array) {

}

TEST(eigen_util, rotate) {
  constexpr int N = 4;
  using Array = eigen_util::FArray<N>;

  Array expected_left_rotate_arrays[N] = {
    {0, 1, 2, 3},
    {1, 2, 3, 0},
    {2, 3, 0, 1},
    {3, 0, 1, 2}
  };

  for (int i = 0; i < N; ++i) {
    Array array{{0, 1, 2, 3}};
    eigen_util::left_rotate(array, i);

    Array expected_array = expected_left_rotate_arrays[i];
    EXPECT_TRUE((array == expected_array).all());
  }

  Array expected_right_rotate_arrays[N] = {
    {0, 1, 2, 3},
    {3, 0, 1, 2},
    {2, 3, 0, 1},
    {1, 2, 3, 0}
  };

  for (int i = 0; i < N; ++i) {
    Array array{{0, 1, 2, 3}};
    eigen_util::right_rotate(array, i);

    Array expected_array = expected_right_rotate_arrays[i];
    EXPECT_TRUE((array == expected_array).all());
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
