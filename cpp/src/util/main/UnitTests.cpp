#include <util/AllocPool.hpp>
#include <util/Random.hpp>

#include <array>
#include <iostream>

int global_pass_count = 0;
int global_fail_count = 0;

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
  bool fail = false;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (orig_a[i]) {
      double pct = counts[i] * 1.0 / N;
      double error = std::abs(pct - 0.5);
      fail |= error > 0.01;
    } else {
      fail |= (counts[i] != 0);
    }
  }

  if (fail) {
    printf("%s<%s> failed!\n", __func__, typeid(T).name());
    for (size_t i = 0; i < counts.size(); ++i) {
      printf("counts[%d]: %d / %d\n", int(i), counts[i], N);
    }
    global_fail_count++;
    return;
  }
  global_pass_count++;
}

void test_random() {
  test_zero_out<bool>();
  test_zero_out<int>();
}

template<typename Pool>
bool test_alloc_pool_helper(Pool& pool, int* sizes, int num_sizes) {
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
  if (int(vec.size()) != x) {
    printf("Expected %d elements, got %lu\n", x, vec.size());
    global_fail_count++;
    return false;
  }

  // validate contents
  for (int i = 0; i < x; ++i) {
    if (vec[i] != i) {
      printf("pool[%d]: expected %d, got %d\n", i, i, vec[i]);
      global_fail_count++;
      return false;
    }
  }

  // TODO: test defrag with contiguous blocks

  // now remove the odd elements
  boost::dynamic_bitset<> used_indices(x);
  for (int i = 1; i < x; i += 2) {
    used_indices[i] = true;
  }
  pool.defragment(used_indices);

  // validate size
  vec = pool.to_vector();
  if (int(vec.size()) != x / 2) {
    printf("Expected %d elements, got %lu\n", x / 2, vec.size());
    global_fail_count++;
    return false;
  }

  // validate contents
  for (int i = 0; i < x / 2; ++i) {
    if (vec[i] != 2 * i + 1) {
      printf("pool[%d]: expected %d, got %d\n", i, 2 * i + 1, vec[i]);
      global_fail_count++;
      return false;
    }
  }

  return true;
}

void test_alloc_pool() {
  using pool_t = util::AllocPool<int, 2>;
  pool_t pool;

  int sizes1[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  if (!test_alloc_pool_helper(pool, sizes1, sizeof(sizes1) / sizeof(sizes1[0]))) return;
  pool.clear();

  int sizes2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  if (!test_alloc_pool_helper(pool, sizes2, sizeof(sizes2) / sizeof(sizes2[0]))) return;
  pool.clear();

  int sizes3[] = {100};
  if (!test_alloc_pool_helper(pool, sizes3, sizeof(sizes3) / sizeof(sizes3[0]))) return;

  global_pass_count++;
}

int main() {
  test_random();
  test_alloc_pool();

  if (global_fail_count > 0) {
    int total_count = global_pass_count + global_fail_count;
    printf("Failed %d of %d test%s!\n", global_fail_count, total_count, total_count > 1 ? "s" : "");
  } else {
    printf("All tests passed (%d of %d)!\n", global_pass_count, global_pass_count);
  }
  return 0;
}