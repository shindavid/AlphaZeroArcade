#include "util/AllocPool.hpp"
#include "util/BoostUtil.hpp"
#include "util/CompactBitSet.hpp"
#include "util/CudaUtil.hpp"
#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/Math.hpp"
#include "util/Random.hpp"
#include "util/StringUtil.hpp"
#include "util/TensorRtUtil.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <array>
#include <map>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

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

TEST(BoostUtil, get_random_set_index) {
  util::Random::set_seed(1);

  int n = 10;
  boost::dynamic_bitset<> bitset(n);
  int index;

  index = boost_util::get_random_set_index(bitset);
  EXPECT_EQ(index, -1);  // No bits are set

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

TEST(Random, zero_out) {
  test_zero_out<bool>();
  test_zero_out<int>();
}

template <typename Pool>
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
    used_indices[i * i] = false;
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
    {3, 10, 500}, {1, 40, 200}, {5, 50, 300}, {4, 20, 400}, {2, 30, 100},
  };

  Array array2 = eigen_util::sort_rows(array);

  Array expected_array2{
    {1, 40, 200}, {2, 30, 100}, {3, 10, 500}, {4, 20, 400}, {5, 50, 300},
  };

  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      EXPECT_EQ(array2(r, c), expected_array2(r, c))
        << " at (" << r << ", " << c << ")"
        << " expected: " << expected_array2(r, c) << " got: " << array2(r, c);
    }
  }

  Array array3 = eigen_util::sort_rows(array, 1, false);

  Array expected_array3{
    {5, 50, 300}, {1, 40, 200}, {2, 30, 100}, {4, 20, 400}, {3, 10, 500},
  };

  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      EXPECT_EQ(array3(r, c), expected_array3(r, c))
        << " at (" << r << ", " << c << ")"
        << " expected: " << expected_array3(r, c) << " got: " << array3(r, c);
    }
  }
}

TEST(eigen_util, UniformDirchletGen) {
  constexpr int M = 1e5;
  constexpr int N = 10;
  float alpha = 0.1;

  eigen_util::UniformDirichletGen<float> gen;
  Eigen::Rand::P8_mt19937_64 rng{35};

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
      EXPECT_NEAR(cov(i, j) / std::sqrt(cov(i, i) * cov(j, j)), expected_cor,
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

TEST(eigen_util, softmax) {
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<N, 1>>;

  Tensor tensor;
  tensor.setValues({{0}, {1}, {2}, {3}});
  Tensor expected;
  expected.setValues({{0.0320586}, {0.0871443}, {0.2368828}, {0.6439143}});

  eigen_util::softmax_in_place(tensor);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(tensor(i, 0), expected(i, 0), 1e-5);
  }
}

TEST(eigen_util, sigmoid) {
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<N, 1>>;

  Tensor tensor;
  tensor.setValues({{0}, {1}, {2}, {3}});
  Tensor expected;
  expected.setValues({{0.5}, {0.7310586}, {0.8807971}, {0.9525741}});

  eigen_util::sigmoid_in_place(tensor);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(tensor(i, 0), expected(i, 0), 1e-5);
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

TEST(eigen_util, all) {
  constexpr int M = 2;
  constexpr int N = 4;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  Tensor tensorHasZero;
  tensorHasZero.setValues({{0, 1, 2, 3}, {4, 5, 6, 7}});
  Tensor tensorAllNonZero;
  tensorAllNonZero.setValues({{1, 2, 3, 4}, {5, 6, 7, 8}});

  EXPECT_TRUE(eigen_util::all(tensorAllNonZero));
  EXPECT_FALSE(eigen_util::all(tensorHasZero));
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
  constexpr int kNumCols = 5;
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

  static std::vector<std::string> column_names = {"ansi", "col1", "col2", "col3", "col4"};

  static eigen_util::PrintArrayFormatMap fmt_map{
    {"ansi", [](float x) { return "\033[32m\u25CF\033[00m"; }},  // green circle
    {"col1", [](float x) { return "foo" + std::to_string((int)x); }},
  };

  std::ostringstream ss;
  eigen_util::print_array(ss, array, column_names, &fmt_map);

  std::string expected_output =
    "ansi col1 col2     col3 col4\n"
    "   \x1B[32m\xE2\x97\x8F\x1B[00m foo0  0.2 0.009275    4\n"
    "   \x1B[32m\xE2\x97\x8F\x1B[00m foo0  0.7 -0.00927    9\n"
    "   \x1B[32m\xE2\x97\x8F\x1B[00m foo1  1.2 1.235e-8   14\n"
    "   \x1B[32m\xE2\x97\x8F\x1B[00m foo1  1.7       18   19\n"
    "   \x1B[32m\xE2\x97\x8F\x1B[00m foo2  2.2       23   24\n"
    "   \x1B[32m\xE2\x97\x8F\x1B[00m foo2  2.7       28   29\n";

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

TEST(eigen_util, extract_rank) {
  using Shape1 = Eigen::Sizes<3, 4, 5>;
  using Shape2 = Eigen::Sizes<2, 3>;

  EXPECT_EQ(eigen_util::extract_rank_v<Shape1>, 3);
  EXPECT_EQ(eigen_util::extract_rank_v<Shape2>, 2);
}

TEST(eigen_util, extract_dim) {
  using Shape1 = Eigen::Sizes<3, 4, 5>;
  using Shape2 = Eigen::Sizes<2, 3>;

  EXPECT_EQ((eigen_util::extract_dim_v<0, Shape1>), 3);
  EXPECT_EQ((eigen_util::extract_dim_v<1, Shape1>), 4);
  EXPECT_EQ((eigen_util::extract_dim_v<2, Shape1>), 5);

  EXPECT_EQ((eigen_util::extract_dim_v<0, Shape2>), 2);
  EXPECT_EQ((eigen_util::extract_dim_v<1, Shape2>), 3);
}

TEST(StringUtil, atof_safe) {
  EXPECT_EQ(util::atof_safe("0.0"), 0.0f);
  EXPECT_EQ(util::atof_safe("5"), 5.0f);
  EXPECT_EQ(util::atof_safe("-5"), -5.0f);
  EXPECT_EQ(util::atof_safe("1.0"), 1.0f);
  EXPECT_EQ(util::atof_safe("3.14"), 3.14f);
  EXPECT_EQ(util::atof_safe("1e-3"), 0.001f);
  EXPECT_EQ(util::atof_safe("1e3"), 1000.0f);
  EXPECT_EQ(util::atof_safe("1.0e3"), 1000.0f);
  EXPECT_EQ(util::atof_safe("1.0e-3"), 0.001f);
  EXPECT_EQ(util::atof_safe("1.0e+3"), 1000.0f);
  EXPECT_EQ(util::atof_safe("-1.0e+3"), -1000.0f);

  // verify that passing non-numeric strings raises an exception
  EXPECT_THROW(util::atof_safe(""), std::invalid_argument);
  EXPECT_THROW(util::atof_safe("abc"), std::invalid_argument);
  EXPECT_THROW(util::atof_safe("1.0abc"), std::invalid_argument);
  EXPECT_THROW(util::atof_safe("1.0eabc"), std::invalid_argument);
}

TEST(StringUtil, split) {
  std::vector<std::string> result1 = util::split("a,b,c", ",");
  std::vector<std::string> result2 = util::split(" a \tb   c ");

  EXPECT_EQ(result1.size(), 3);
  EXPECT_EQ(result1[0], "a");
  EXPECT_EQ(result1[1], "b");
  EXPECT_EQ(result1[2], "c");

  EXPECT_EQ(result2.size(), 3);
  EXPECT_EQ(result2[0], "a");
  EXPECT_EQ(result2[1], "b");
  EXPECT_EQ(result2[2], "c");

  std::vector<std::string> result3;
  int n3;
  n3 = util::split(result3, "a,bb,ccc", ",");

  EXPECT_EQ(n3, 3);
  EXPECT_EQ(result3.size(), 3);
  EXPECT_EQ(result3[0], "a");
  EXPECT_EQ(result3[1], "bb");
  EXPECT_EQ(result3[2], "ccc");

  n3 = util::split(result3, "\t\taa  b\n");

  // vector reuse - leave 3rd element unchanged!
  EXPECT_EQ(n3, 2);
  EXPECT_EQ(result3.size(), 3);
  EXPECT_EQ(result3[0], "aa");
  EXPECT_EQ(result3[1], "b");
  EXPECT_EQ(result3[2], "ccc");

  n3 = util::split(result3, "\t\taa  b c    d  e\n");

  EXPECT_EQ(n3, 5);
  EXPECT_EQ(result3.size(), 5);
  EXPECT_EQ(result3[0], "aa");
  EXPECT_EQ(result3[1], "b");
  EXPECT_EQ(result3[2], "c");
  EXPECT_EQ(result3[3], "d");
  EXPECT_EQ(result3[4], "e");
}

TEST(StringUtil, splitlines) {
  std::vector<std::string> result1 = util::splitlines("a\nb\nc");
  std::vector<std::string> result2 = util::splitlines("\n ac\n");
  std::vector<std::string> result3 = util::splitlines("");
  std::vector<std::string> result4 = util::splitlines("\n");

  EXPECT_EQ(result1.size(), 3);
  EXPECT_EQ(result1[0], "a");
  EXPECT_EQ(result1[1], "b");
  EXPECT_EQ(result1[2], "c");

  EXPECT_EQ(result2.size(), 2);
  EXPECT_EQ(result2[0], "");
  EXPECT_EQ(result2[1], " ac");

  EXPECT_EQ(result3.size(), 0);

  EXPECT_EQ(result4.size(), 1);
  EXPECT_EQ(result4[0], "");
}

TEST(StringUtil, ends_with) {
  EXPECT_TRUE(util::ends_with("hello", "lo"));
  EXPECT_TRUE(util::ends_with("hello", "hello"));
  EXPECT_FALSE(util::ends_with("hello", "hello world"));
  EXPECT_FALSE(util::ends_with("hello", "hello hello"));
  EXPECT_TRUE(util::ends_with("hello", ""));
  EXPECT_TRUE(util::ends_with("", ""));
  EXPECT_FALSE(util::ends_with("", "a"));
}

TEST(StringUtil, terminal_width) {
  EXPECT_EQ(util::terminal_width(""), 0);
  EXPECT_EQ(util::terminal_width("hello"), 5);
  EXPECT_EQ(util::terminal_width("\033[31m\033[00m"), 0);       // red font
  EXPECT_EQ(util::terminal_width("\033[31mhello\033[00m"), 5);  // red font
}

TEST(StringUtil, grammatically_join) {
  EXPECT_EQ(util::grammatically_join({"a"}, "and"), "a");
  EXPECT_EQ(util::grammatically_join({"a", "b"}, "and"), "a and b");
  EXPECT_EQ(util::grammatically_join({"a", "b"}, "and", false), "a and b");
  EXPECT_EQ(util::grammatically_join({"a", "b", "c"}, "or"), "a, b, or c");
  EXPECT_EQ(util::grammatically_join({"a", "b", "c"}, "or", false), "a, b or c");
}

TEST(StringUtil, parse_bytes) {
  EXPECT_EQ(util::parse_bytes("256MiB"), 256ull << 20);
  EXPECT_EQ(util::parse_bytes("256MB"), 256ull * 1000 * 1000);
  EXPECT_EQ(util::parse_bytes("1GiB"), 1ull << 30);
  EXPECT_EQ(util::parse_bytes("1.5GiB"), 1.5 * (1ull << 30));
  EXPECT_EQ(util::parse_bytes("1073741824"), 1073741824ull);
}

TEST(cuda_util, cuda_device_to_ordinal) {
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("cuda:0"), 0);
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("0"), 0);
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("cuda:1"), 1);
  EXPECT_EQ(cuda_util::cuda_device_to_ordinal("1"), 1);
}

TEST(math, normal_cdf) {
  EXPECT_NEAR(math::normal_cdf(0), 0.5, 1e-6);
  EXPECT_NEAR(math::normal_cdf(1), 0.8413447, 1e-6);
  EXPECT_NEAR(math::normal_cdf(-1), 0.1586553, 1e-6);
  EXPECT_NEAR(math::normal_cdf(2), 0.9772499, 1e-6);
  EXPECT_NEAR(math::normal_cdf(-2), 0.0227501, 1e-6);
}

TEST(math, splitmix64) {
  constexpr int log2_num_buckets = 3;
  constexpr int num_buckets = 1 << log2_num_buckets;
  constexpr int N = 100000;
  int counts[num_buckets] = {};

  for (int i = 0; i < N; ++i) {
    uint64_t h = math::splitmix64(i);
    int bucket = h >> (64 - log2_num_buckets);
    counts[bucket]++;
  }
  for (int i = 0; i < num_buckets; ++i) {
    double pct = counts[i] * 1.0 / N;
    EXPECT_NEAR(pct, 1.0 / num_buckets, 0.01);
  }
}

TEST(TensorRtUtil, parse_precision) {
  EXPECT_EQ(trt_util::parse_precision("FP16"), trt_util::Precision::kFP16);
  EXPECT_EQ(trt_util::parse_precision("FP32"), trt_util::Precision::kFP32);
  EXPECT_EQ(trt_util::parse_precision("INT8"), trt_util::Precision::kINT8);

  EXPECT_THROW(trt_util::parse_precision("FP64"), util::CleanException);
}

TEST(TensorRtUtil, get_version_tag) {
  const char* version_tag = trt_util::get_version_tag();
  EXPECT_STREQ(version_tag, "10.11.0");
}

/////////////////////////////
// Tests for CompactBitSet //
/////////////////////////////

// helper: collect a range of indices into a vector<int>
template <class Range>
std::vector<int> collect(Range&& r) {
  std::vector<int> v;
  for (auto i : r) v.push_back(static_cast<int>(i));
  return v;
}

// Helper: bounds checker for “roughly uniform” counts.
// Ensures each count is within +/-tol_frac of expectation.
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

// --------------------- One-word cases (N <= 64 by your layout) ---------------------

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
  // set a few bits
  bs.set(0).set(5).set(11);
  EXPECT_TRUE(bs.test(0));
  EXPECT_TRUE(bs[5]);
  EXPECT_TRUE(bs.test(11));
  EXPECT_EQ(bs.count(), 3u);

  // reset a bit
  bs.reset(5);
  EXPECT_FALSE(bs.test(5));
  EXPECT_EQ(bs.count(), 2u);

  // set all / reset all
  bs.set();
  EXPECT_TRUE(bs.all());
  EXPECT_EQ(bs.count(), BS::size());
  bs.reset();
  EXPECT_TRUE(bs.none());
}

TEST(CompactBitSet, BitwiseOps_OneWord) {
  using BS = util::CompactBitSet<16>;
  BS a, b;

  // a = 0001 0010 0000 0001 (bits 0,5,12)
  a.set(0).set(5).set(12);
  // b = 0000 0010 0001 0001 (bits 0,4,12)
  b.set(0).set(4).set(12);

  BS c_and = a & b;  // bits 0,12
  BS c_or = a | b;   // bits 0,4,5,12
  BS c_xor = a ^ b;  // bits 4,5

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

  // ~ obeys N (tail masked)
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
  // set bits: 0,2,3,10,15
  bs.set(0).set(2).set(3).set(10).set(15);

  // get_nth_on_index (within range)
  EXPECT_EQ(bs.get_nth_on_index(0), 0);
  EXPECT_EQ(bs.get_nth_on_index(1), 2);
  EXPECT_EQ(bs.get_nth_on_index(2), 3);
  EXPECT_EQ(bs.get_nth_on_index(3), 10);
  EXPECT_EQ(bs.get_nth_on_index(4), 15);

  // count_on_indices_before
  EXPECT_EQ(bs.count_on_indices_before(0), 0);
  EXPECT_EQ(bs.count_on_indices_before(1), 1);   // only bit 0
  EXPECT_EQ(bs.count_on_indices_before(2), 1);   // still only bit 0
  EXPECT_EQ(bs.count_on_indices_before(3), 2);   // bits 0,2
  EXPECT_EQ(bs.count_on_indices_before(4), 3);   // bits 0,2,3
  EXPECT_EQ(bs.count_on_indices_before(11), 4);  // + bit 10
  EXPECT_EQ(bs.count_on_indices_before(20), 5);  // all five
}

TEST(CompactBitSet, ToStringNatural_OneWord) {
  using BS = util::CompactBitSet<8>;
  BS bs;
  bs.set(0).set(3).set(7);
  // s[k] is bit k ('0'/'1')
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
  // Cross the 64-bit boundary
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
  BS all = ~zero;  // should set exactly 70 bits
  EXPECT_TRUE(all.any());
  EXPECT_TRUE(all.all());
  EXPECT_EQ(all.count(), 70u);

  BS bs;
  bs.set(0).set(64).set(69);
  BS mask;
  mask.set();  // all ones in the logical 70-bit domain
  BS anded = bs & mask;
  EXPECT_EQ(anded.count(), 3u);
  EXPECT_TRUE(anded.test(0));
  EXPECT_TRUE(anded.test(64));
  EXPECT_TRUE(anded.test(69));

  // |= and ^= should not disturb tail if writers keep the invariant
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
  // Set a few on both sides of the boundary
  bs.set(0).set(3).set(4).set(64).set(66).set(69);

  auto on = collect(bs.on_indices());
  auto off = collect(bs.off_indices());

  std::vector<int> want_on = {0, 3, 4, 64, 66, 69};
  EXPECT_EQ(on, want_on);

  // spot-check OFF contains boundary-adjacent indices and last few
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
  // Put some bits across both words
  // ON at: 0, 2, 3, 10, 64, 66, 69
  bs.set(0).set(2).set(3).set(10).set(64).set(66).set(69);

  // rank-select
  EXPECT_EQ(bs.get_nth_on_index(0), 0);
  EXPECT_EQ(bs.get_nth_on_index(1), 2);
  EXPECT_EQ(bs.get_nth_on_index(2), 3);
  EXPECT_EQ(bs.get_nth_on_index(3), 10);
  EXPECT_EQ(bs.get_nth_on_index(4), 64);
  EXPECT_EQ(bs.get_nth_on_index(5), 66);
  EXPECT_EQ(bs.get_nth_on_index(6), 69);

  // prefix counts
  EXPECT_EQ(bs.count_on_indices_before(0), 0);
  EXPECT_EQ(bs.count_on_indices_before(1), 1);   // bit 0
  EXPECT_EQ(bs.count_on_indices_before(4), 3);   // 0,2,3
  EXPECT_EQ(bs.count_on_indices_before(11), 4);  // +10
  EXPECT_EQ(bs.count_on_indices_before(64), 4);  // none across boundary yet
  EXPECT_EQ(bs.count_on_indices_before(65), 5);  // +64
  EXPECT_EQ(bs.count_on_indices_before(70), 7);  // all seven
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

// Random methods of CompactBitSet

// --------------------- choose_random_on_index ---------------------

TEST(CompactBitSet_Random, ChooseRandomOnIndex_OneWord_BasicAndUniform) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<20>;
  BS bs;
  // ON at: 0, 3, 7, 12, 15
  std::vector<int> on = {0, 3, 7, 12, 15};
  for (int k : on) bs.set(k);

  // Basic sanity: always returns an ON index in range.
  const int trials = 20000;
  std::vector<int> hist(on.size(), 0);

  for (int t = 0; t < trials; ++t) {
    int idx = bs.choose_random_on_index();
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, static_cast<int>(BS::size()));
    ASSERT_TRUE(bs.test(static_cast<size_t>(idx)));

    // bucketize
    auto it = std::find(on.begin(), on.end(), idx);
    ASSERT_NE(it, on.end());
    ++hist[static_cast<int>(std::distance(on.begin(), it))];
  }

  // Rough uniformity check (±12%)
  ExpectRoughlyUniform(hist, trials, 0.12);
}

TEST(CompactBitSet_Random, ChooseRandomOnIndex_MultiWord_BasicAndUniform) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<70>;
  BS bs;
  // ON spanning the 64-boundary: 0, 2, 3, 10, 64, 66, 69
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

  // Rough uniformity check (±10%)
  ExpectRoughlyUniform(hist, trials, 0.10);
}

// --------------------- randomly_zero_out ---------------------

TEST(CompactBitSet_Random, RandomlyZeroOut_OneWord_CountAndMembership) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<24>;
  BS bs;
  // ON at 0..15 (16 bits on)
  for (int i = 0; i < 16; ++i) bs.set(i);

  // Copy to capture original ON set
  BS original = bs;

  // Remove 5 at random
  bs.randomly_zero_out(5);

  // Count decreased by 5
  EXPECT_EQ(bs.count(), original.count() - 5);

  // The cleared bits are a subset of the original ON set
  int cleared = 0;
  for (int i = 0; i < 16; ++i) {
    if (original.test(i) && !bs.test(i)) ++cleared;
  }
  EXPECT_EQ(cleared, 5);
}

TEST(CompactBitSet_Random, RandomlyZeroOut_OneWord_UniformSingles) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<20>;
  // Initial ON at: 0..9 (10 choices)
  std::vector<int> base_on(10);
  std::iota(base_on.begin(), base_on.end(), 0);

  const int choices = static_cast<int>(base_on.size());
  const int trials = 20000;
  std::vector<int> hist(choices, 0);

  for (int t = 0; t < trials; ++t) {
    BS bs;
    for (int k : base_on) bs.set(k);

    // Zero out exactly one bit at random
    bs.randomly_zero_out(1);

    // Find which bit was cleared
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

  // Rough uniformity (±12%)
  ExpectRoughlyUniform(hist, trials, 0.12);
}

TEST(CompactBitSet_Random, RandomlyZeroOut_MultiWord_CountAndUniformSingles) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<70>;
  // ON at: 0..31 and 64..69 (total 38 choices)
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

    // Remove 1 at random
    bs.randomly_zero_out(1);

    // Count decreased by 1
    EXPECT_EQ(bs.count(), original.count() - 1);

    // Identify which bit cleared
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

  // Rough uniformity across all 38 choices (±10%)
  ExpectRoughlyUniform(hist, trials, 0.10);
}

TEST(CompactBitSet_Random, RandomlyZeroOut_AllBits) {
  util::Random::set_seed(1);

  using BS = util::CompactBitSet<33>;  // crosses 32->64 word boundary policy
  BS bs;
  for (size_t i = 0; i < BS::size(); ++i) bs.set(i);
  const auto before = bs.count();

  bs.randomly_zero_out(static_cast<int>(before));
  EXPECT_TRUE(bs.none());
  EXPECT_EQ(bs.count(), 0u);
}

/////////////////////////////////
// End tests for CompactBitSet //
/////////////////////////////////

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
