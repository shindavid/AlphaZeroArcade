#include <util/Random.hpp>

#include <array>
#include <iostream>

int global_pass_count = 0;
int global_fail_count = 0;

template<typename T>
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

int main() {
  test_random();

  if (global_fail_count > 0) {
    int total_count = global_pass_count + global_fail_count;
    printf("Failed %d of %d test%s!\n", global_fail_count, total_count, total_count > 1 ? "s" : "");
  } else {
    printf("All tests passed (%d of %d)!\n", global_pass_count, global_pass_count);
  }
  return 0;
}