#include "util/AllocPool.hpp"
#include "util/GTestUtil.hpp"

#include <boost/dynamic_bitset.hpp>
#include <gtest/gtest.h>

#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

namespace {

template <typename Pool>
void test_alloc_pool_helper(Pool& pool, int* sizes, int num_sizes) {
  int x = 0;
  for (int i = 0; i < num_sizes; ++i) {
    int size = sizes[i];
    util::pool_index_t idx = pool.alloc(size);
    for (int j = 0; j < size; ++j) {
      pool[idx + j] = x++;
    }
  }

  std::vector<int> vec = pool.to_vector();
  EXPECT_EQ(vec.size(), x);

  for (int i = 0; i < x; ++i) {
    EXPECT_EQ(vec[i], i);
  }

  boost::dynamic_bitset<> used_indices(x);
  used_indices.set();
  int y = x;
  for (int i = 0; i * i < x; ++i) {
    used_indices[i * i] = false;
    --y;
  }
  pool.defragment(used_indices);

  vec = pool.to_vector();
  EXPECT_EQ(vec.size(), y);

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

}  // namespace

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

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
