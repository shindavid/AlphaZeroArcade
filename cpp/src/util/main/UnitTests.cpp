#include <util/AllocPool.hpp>
#include <util/EigenUtil.hpp>
#include <util/Random.hpp>
#include <util/StringUtil.hpp>

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
    for (int j = 0; j < size; ++j) {
      pool[idx + j] = x++;
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

TEST(eigen_util, sort_rows) {
  constexpr int kNumRows = 5;
  constexpr int kNumCols = 3;
  constexpr int kMaxNumRows = 10;
  using Array = Eigen::Array<float, Eigen::Dynamic, kNumCols, 0, kMaxNumRows, kNumCols>;

  Array array{
      {3, 30, 300},
      {1, 10, 100},
      {5, 50, 500},
      {4, 40, 400},
      {2, 20, 200},
  };

  array = eigen_util::sort_rows(array);

  int pow10[] = {1, 10, 100};
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      float expected = (r + 1) * pow10[c];
      EXPECT_EQ(array(r, c), expected);
    }
  }
}

TEST(eigen_util, UniformDirchletGen) {
  constexpr int M = 1e5;
  constexpr int N = 10;
  float alpha = 0.1;

  eigen_util::UniformDirichletGen<float> gen;
  Eigen::Rand::P8_mt19937_64 rng{ 35 };

  // dynamic size matrix here due to the size of sample matrix X
  Eigen::MatrixXf X(M, N);
  X.setZero();

  for (int i = 0; i < M; ++i) {
    X.row(i) = gen.template generate<Eigen::Array<float, N, 1>>(rng, alpha).transpose();
  }

  Eigen::MatrixXf cov = eigen_util::compute_covariance(X);

  // wikipedia formula for Dirichlet moments:
  // https://en.wikipedia.org/wiki/Dirichlet_distribution
  float expected_var = 1.0 / N * (1 - 1.0 / N) / (alpha * N + 1);
  float expected_cor = -1.0 / (N - 1);

  // check the variances
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(cov(i, i), expected_var, std::abs(expected_var) * 0.05);
  }

  // check the correlations
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < j; i++) {
      EXPECT_NEAR(cov(i, j)/std::sqrt(cov(i, i) * cov(j, j)), expected_cor,
      std::abs(expected_cor) * 0.05);
    }
  }

  // check mean convergence with M according to Central Limit Theorem
  // 99% confidence interval
  Eigen::VectorXf mean = X.colwise().mean();
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(mean(i), 1.0 / N, 2.576 * std::sqrt(expected_var / M));
  }
}

TEST(eigen_util, sort_columns_one_element) {
  constexpr int kNumRows = 1;
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

TEST(eigen_util, softmax_Array) {
  constexpr int N = 4;
  using Array = eigen_util::FArray<N>;

  Array array{{0, 1, 2, 3}};
  Array expected{{0.0320586, 0.0871443, 0.2368828, 0.6439143}};

  array = eigen_util::softmax(array);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(array(i), expected(i), 1e-5);
  }
}

TEST(eigen_util, softmax_Tensor) {
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<N, 1>>;

  Tensor tensor;
  tensor.setValues({{0}, {1}, {2}, {3}});
  Tensor expected;
  expected.setValues({{0.0320586}, {0.0871443}, {0.2368828}, {0.6439143}});

  tensor = eigen_util::softmax(tensor);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(tensor(i, 0), expected(i, 0), 1e-5);
  }
}

TEST(eigen_util, sigmoid) {
  constexpr int N = 4;
  using Array = eigen_util::FArray<N>;

  Array array{{0, 1, 2, 3}};
  Array expected{{0.5, 0.7310586, 0.8807971, 0.9525741}};

  array = eigen_util::sigmoid(array);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(array(i), expected(i), 1e-5);
  }
}

TEST(eigen_util, reverse) {
  constexpr int M = 3;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<3, 4>>;

  Tensor tensor;
  tensor.setValues({{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
  Tensor expected;
  expected.setValues({{3, 2, 1, 0}, {7, 6, 5, 4}, {11, 10, 9, 8}});

  Tensor reversed = eigen_util::reverse(tensor, 1);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(reversed(i, j), expected(i, j));
    }
  }
}

TEST(eigen_util, normalize) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;
  using Array = Eigen::Array<float, M, N, Eigen::RowMajor>;

  Array array{{0, 1, 2, 3}, {0, 0, 4, 0}};
  Array expected = array / array.sum();
  Tensor tensor = Eigen::TensorMap<Tensor>(array.data(), array.rows(), array.cols());

  bool success = eigen_util::normalize(tensor, 1e-5);

  EXPECT_TRUE(success);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_NEAR(tensor(i, j), expected(i, j), 1e-5);
    }
  }
}

TEST(eigen_util, sample) {
  util::Random::set_seed(1);

  constexpr int N = 4;
  constexpr int numSamples = 10000;

  using Tensor = eigen_util::FTensor<Eigen::Sizes<N>>;

  Tensor values;
  values.setValues({1, 2, 3, 4});
  Tensor expectedFreq = values * (numSamples / eigen_util::sum(values));
 // *numSamples;

  Tensor freq;
  freq.setZero();
  for (int i = 0; i < numSamples; i++) {
    auto sample = eigen_util::sample(values);
    freq(sample)++;
  }

  // Chi-Squared Statistic \sum{(O_i - E_i)^2 / E_i}
  // where O_i is the observed frequency and E_i is the expected frequency
  float chi2 = 0;
  for (int i = 0; i < N; i++) {
    float diff = freq(i) - expectedFreq(i);
    chi2 += diff * diff / expectedFreq(i);
  }

  // The chi-squared value corresponding to a p-value of 0.05 with 3 degrees of freedom is 7.81
  EXPECT_LT(chi2, 7.81);
}

TEST(eigen_util, randomly_zero_out) {
  util::Random::set_seed(1);

  constexpr int M = 8;
  constexpr int N = 4;
  constexpr int numZeroes = 10;
  static_assert(numZeroes <= M * N, "numZeroes must be less than or equal to M * N");

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;
  Tensor tensor;
  tensor.setConstant(1);

  eigen_util::randomly_zero_out(tensor, numZeroes);

  auto data = tensor.data();
  auto size = tensor.size();
  int zero_count = 0;

  for (int i = 0; i < size; ++i) {
    if (data[i] == 0) {
      zero_count++;
    }
  }
  EXPECT_EQ(zero_count, numZeroes);
}

TEST(eigen_util, reinterpret_as_array) {
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<N, 1>>;
  using Array = eigen_util::FArray<N>;

  Tensor tensor;
  tensor.setValues({{0}, {1}, {2}, {3}});
  Array expected_array = {{0, 1, 2, 3}};

  auto array = eigen_util::reinterpret_as_array(tensor);

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(array(i, 0), expected_array(i, 0));
  }
}

TEST(eigen_util, cwiseMax) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;
  using Array = Eigen::Array<float, M, N, Eigen::RowMajor>;

  Tensor tensor;
  tensor.setValues({{0, 1, 2, 3}, {4, 5, 6, 7}});
  Array expected = {{2, 2, 2, 3}, {4, 5, 6, 7}};

  auto result = eigen_util::cwiseMax(tensor, 2);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }
}

TEST(eigen_util, assert_is_valid_prob_distr_array) {
  constexpr int N = 4;
  using Array = eigen_util::FArray<N>;

  Array valid{{0.1, 0.2, 0.3, 0.4}};
  EXPECT_NO_THROW(eigen_util::assert_is_valid_prob_distr(valid, 1e-5));

  Array invalid{{0.1, 0.2, 0.3, 0.5}};
  EXPECT_ANY_THROW(eigen_util::assert_is_valid_prob_distr(invalid, 1e-5));
}

TEST(eigen_util, assert_is_valid_prob_distr_tensor) {
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<N, 1>>;

  Tensor valid;
  valid.setValues({{0.1}, {0.2}, {0.3}, {0.4}});
  EXPECT_NO_THROW(eigen_util::assert_is_valid_prob_distr(valid, 1e-5));

  Tensor invalid;
  invalid.setValues({{0.1}, {0.2}, {0.3}, {0.5}});
  EXPECT_ANY_THROW(eigen_util::assert_is_valid_prob_distr(invalid, 1e-5));
}

TEST(eigen_util, sum) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  Tensor tensor;
  tensor.setValues({{0, 1, 2, 3}, {4, 5, 6, 7}});
  float expected = 28;

  float result = eigen_util::sum(tensor);
  EXPECT_EQ(result, expected);
}

TEST(eigen_util, max) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  Tensor tensor;
  tensor.setValues({{0, 1, 2, 3}, {4, 5, 6, 7}});
  float expected = 7;

  float result = eigen_util::max(tensor);
  EXPECT_EQ(result, expected);
}

TEST(eigen_util, min) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  Tensor tensor;
  tensor.setValues({{0, 1, 2, 3}, {4, 5, 6, 7}});
  float expected = 0;

  float result = eigen_util::min(tensor);
  EXPECT_EQ(result, expected);
}

TEST(eigen_util, any) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  Tensor tensorHasNonZero;
  tensorHasNonZero.setValues({{0, 1, 2, 3}, {4, 5, 6, 7}});
  Tensor tensorAllZero;
  tensorAllZero.setValues({{0, 0, 0, 0}, {0, 0, 0, 0}});

  EXPECT_TRUE(eigen_util::any(tensorHasNonZero));
  EXPECT_FALSE(eigen_util::any(tensorAllZero));
}

TEST(eigen_util, count) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  Tensor tensor;
  tensor.setValues({{0, 1, 2, 3}, {4, 5, 6, 0}});
  int expected = 6;

  int result = eigen_util::count(tensor);
  EXPECT_EQ(result, expected);
}

TEST(eigen_util, rotate) {
  constexpr int N = 4;
  using Array = eigen_util::FArray<N>;

  Array expected_left_rotate_arrays[N] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}};

  for (int i = 0; i < N; ++i) {
    Array array{{0, 1, 2, 3}};
    eigen_util::left_rotate(array, i);

    Array expected_array = expected_left_rotate_arrays[i];
    EXPECT_TRUE((array == expected_array).all());
  }

  Array expected_right_rotate_arrays[N] = {{0, 1, 2, 3}, {3, 0, 1, 2}, {2, 3, 0, 1}, {1, 2, 3, 0}};

  for (int i = 0; i < N; ++i) {
    Array array{{0, 1, 2, 3}};
    eigen_util::right_rotate(array, i);

    Array expected_array = expected_right_rotate_arrays[i];
    EXPECT_TRUE((array == expected_array).all());
  }
}

TEST(eigen_util, print_array) {
  constexpr int kNumRows = 6;
  constexpr int kNumCols = 4;
  constexpr int kMaxRows = 10;
  using Array = Eigen::Array<float, Eigen::Dynamic, kNumCols, 0, kMaxRows, kNumCols>;

  Array array(kNumRows, kNumCols);

  // Fill the array with values
  for (int i = 0; i < kNumRows; ++i) {
    for (int j = 0; j < kNumCols; ++j) {
      array(i, j) = i * kNumCols + j;
    }
  }

  array.col(1) /= 10;
  array.col(2) /= 10;
  array(0, 3) = 0.00927465;
  array(1, 3) = -0.00927465;
  array(2, 3) = 1.23456e-8;

  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};

  eigen_util::PrintArrayFormatMap fmt_map;
  fmt_map["col1"] = [](float x) { return "foo" + std::to_string((int)x); };

  std::ostringstream ss;
  eigen_util::print_array(ss, array, column_names, &fmt_map);

  std::string expected_output =
      " col1 col2 col3     col4\n"
      " foo0  0.1  0.2 0.009275\n"
      " foo4  0.5  0.6 -0.00927\n"
      " foo8  0.9    1 1.235e-8\n"
      "foo12  1.3  1.4       15\n"
      "foo16  1.7  1.8       19\n"
      "foo20  2.1  2.2       23\n";

  EXPECT_EQ(ss.str(), expected_output);
}

TEST(eigen_util, concatenate_columns) {
  using Array1 = Eigen::Array<float, 4, 1>;
  using Array2 = Eigen::Array<float, 4, 3>;

  Array1 a{1, 2, 3, 4};
  Array1 b{5, 6, 7, 8};
  Array1 c{9, 10, 11, 12};

  Array2 expected = {{1, 5, 9}, {2, 6, 10}, {3, 7, 11}, {4, 8, 12}};

  Array2 actual = eigen_util::concatenate_columns(a, b, c);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(actual(i, j), expected(i, j));
    }
  }
}

TEST(StringUtil, terminal_width) {
  EXPECT_EQ(util::terminal_width(""), 0);
  EXPECT_EQ(util::terminal_width("hello"), 5);
  EXPECT_EQ(util::terminal_width("\033[31m\033[00m"), 0);  // red font
  EXPECT_EQ(util::terminal_width("\033[31mhello\033[00m"), 5);  // red font
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}