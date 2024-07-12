#include <util/AllocPool.hpp>
#include <util/EigenUtil.hpp>
#include <util/Random.hpp>

#include <Eigen/Core>

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
  if (int(vec.size()) != y) {
    printf("Expected %d elements, got %lu\n", y, vec.size());
    global_fail_count++;
    return false;
  }

  // validate contents
  int sqrt = 0;
  int k = 0;
  for (int i = 0; i < x; ++i) {
    if (sqrt * sqrt == i) {
      ++sqrt;
      continue;
    }
    if (vec[k] != i) {
      printf("pool[%d]: expected %d, got %d\n", k, i, vec[i]);
      global_fail_count++;
      return false;
    }
    ++k;
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

void test_eigen_util() {
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
      if (array(r, c) != expected) {
        printf("%s() failure at %s:%d\n", __func__, __FILE__, __LINE__);
        printf("Expected %.f at array(%d, %d) but got %.f\n", expected, r, c, array(r, c));
        std::cout << array << std::endl;
        global_fail_count++;
        return;
      }
    }
  }

  global_pass_count++;
}

int main() {
  test_random();
  test_alloc_pool();
  test_eigen_util();

  if (global_fail_count > 0) {
    int total_count = global_pass_count + global_fail_count;
    printf("Failed %d of %d test%s!\n", global_fail_count, total_count, total_count > 1 ? "s" : "");
  } else {
    printf("All tests passed (%d of %d)!\n", global_pass_count, global_pass_count);
  }
  return 0;
}