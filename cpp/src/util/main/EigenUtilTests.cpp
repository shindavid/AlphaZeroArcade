#include "util/EigenUtil.hpp"
#include "util/GTestUtil.hpp"
#include "util/Random.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

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

TEST(eigen_util, softmax_in_place_array) {
  constexpr int N = 4;
  using Array = eigen_util::FArray<N>;

  Array array{0, 1, 2, 3};
  Array expected{0.0320586, 0.0871443, 0.2368828, 0.6439143};

  eigen_util::softmax_in_place(array);

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(array(i), expected(i), 1e-5);
  }
}

TEST(eigen_util, softmax_in_place_tensor) {
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

TEST(eigen_util, rowwise_softmax_in_place) {
  {
    constexpr int M = 2;
    constexpr int N = 4;
    using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

    Tensor tensor;
    tensor.setValues({{0, 1, 2, 3}, {1, 1, 1, 1}});
    Tensor expected;
    expected.setValues({{0.0320586, 0.0871443, 0.2368828, 0.6439143}, {0.25, 0.25, 0.25, 0.25}});

    eigen_util::rowwise_softmax_in_place(tensor);

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        EXPECT_NEAR(tensor(i, j), expected(i, j), 1e-5);
      }
    }
  }
  {
    constexpr int M = 1;
    constexpr int N = 2;
    using Tensor = eigen_util::FTensor<Eigen::Sizes<M, N>>;

    Tensor tensor;
    tensor.setValues({{2, 0}});
    Tensor expected;
    expected.setValues({{0.8807971, 0.1192029}});

    eigen_util::rowwise_softmax_in_place(tensor);

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        EXPECT_NEAR(tensor(i, j), expected(i, j), 1e-5);
      }
    }
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

TEST(eigen_util, rot90_clockwise) {
  constexpr int M = 10;
  constexpr int N = 2;
  using Tensor2D = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  // 0 1 2      6 3 0
  // 3 4 5   -> 7 4 1
  // 6 7 8      8 5 2
  Tensor2D tensor_2d;
  tensor_2d.setValues(
    {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}, {8, 18}, {9, 19}});

  Tensor2D expected_2d;
  expected_2d.setValues(
    {{6, 16}, {3, 13}, {0, 10}, {7, 17}, {4, 14}, {1, 11}, {8, 18}, {5, 15}, {2, 12}, {9, 19}});

  eigen_util::rot90_clockwise<3>(tensor_2d);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(tensor_2d(i, j), expected_2d(i, j)) << "tensor_2d:\n"
                                                    << tensor_2d << "\nexpected_2d:\n"
                                                    << expected_2d;
    }
  }

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M>>;
  Tensor tensor;
  tensor.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor expected;
  expected.setValues({6, 3, 0, 7, 4, 1, 8, 5, 2, 9});
  eigen_util::rot90_clockwise<3>(tensor);

  for (int i = 0; i < M; ++i) {
    EXPECT_EQ(tensor(i), expected(i)) << "tensor:\n" << tensor << "\nexpected:\n" << expected;
  }
}

TEST(eigen_util, rot180) {
  constexpr int M = 10;
  constexpr int N = 2;
  using Tensor2D = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  // 0 1 2      8 7 6
  // 3 4 5   -> 5 4 3
  // 6 7 8      2 1 0
  Tensor2D tensor_2d;
  tensor_2d.setValues(
    {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}, {8, 18}, {9, 19}});

  Tensor2D expected_2d;
  expected_2d.setValues(
    {{8, 18}, {7, 17}, {6, 16}, {5, 15}, {4, 14}, {3, 13}, {2, 12}, {1, 11}, {0, 10}, {9, 19}});

  eigen_util::rot180<3>(tensor_2d);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(tensor_2d(i, j), expected_2d(i, j)) << "tensor_2d:\n"
                                                    << tensor_2d << "\nexpected_2d:\n"
                                                    << expected_2d;
    }
  }

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M>>;
  Tensor tensor;
  tensor.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor expected;
  expected.setValues({8, 7, 6, 5, 4, 3, 2, 1, 0, 9});
  eigen_util::rot180<3>(tensor);

  for (int i = 0; i < M; ++i) {
    EXPECT_EQ(tensor(i), expected(i)) << "tensor:\n" << tensor << "\nexpected:\n" << expected;
  }
}

TEST(eigen_util, rot270_clockwise) {
  constexpr int M = 10;
  constexpr int N = 2;
  using Tensor2D = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  // 0 1 2      2 5 8
  // 3 4 5   -> 1 4 7
  // 6 7 8      0 3 6
  Tensor2D tensor_2d;
  tensor_2d.setValues(
    {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}, {8, 18}, {9, 19}});

  Tensor2D expected_2d;
  expected_2d.setValues(
    {{2, 12}, {5, 15}, {8, 18}, {1, 11}, {4, 14}, {7, 17}, {0, 10}, {3, 13}, {6, 16}, {9, 19}});

  eigen_util::rot270_clockwise<3>(tensor_2d);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(tensor_2d(i, j), expected_2d(i, j)) << "tensor_2d:\n"
                                                    << tensor_2d << "\nexpected_2d:\n"
                                                    << expected_2d;
    }
  }

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M>>;
  Tensor tensor;
  tensor.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor expected;
  expected.setValues({2, 5, 8, 1, 4, 7, 0, 3, 6, 9});
  eigen_util::rot270_clockwise<3>(tensor);

  for (int i = 0; i < M; ++i) {
    EXPECT_EQ(tensor(i), expected(i)) << "tensor:\n" << tensor << "\nexpected:\n" << expected;
  }
}

TEST(eigen_util, flip_vertical) {
  constexpr int M = 10;
  constexpr int N = 2;
  using Tensor2D = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  // 0 1 2      6 7 8
  // 3 4 5   -> 3 4 5
  // 6 7 8      0 1 2
  Tensor2D tensor_2d;
  tensor_2d.setValues(
    {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}, {8, 18}, {9, 19}});

  Tensor2D expected_2d;
  expected_2d.setValues(
    {{6, 16}, {7, 17}, {8, 18}, {3, 13}, {4, 14}, {5, 15}, {0, 10}, {1, 11}, {2, 12}, {9, 19}});

  eigen_util::flip_vertical<3>(tensor_2d);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(tensor_2d(i, j), expected_2d(i, j)) << "tensor_2d:\n"
                                                    << tensor_2d << "\nexpected_2d:\n"
                                                    << expected_2d;
    }
  }

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M>>;
  Tensor tensor;
  tensor.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor expected;
  expected.setValues({6, 7, 8, 3, 4, 5, 0, 1, 2, 9});
  eigen_util::flip_vertical<3>(tensor);

  for (int i = 0; i < M; ++i) {
    EXPECT_EQ(tensor(i), expected(i)) << "tensor:\n" << tensor << "\nexpected:\n" << expected;
  }
}

TEST(eigen_util, mirror_horizontal) {
  constexpr int M = 10;
  constexpr int N = 2;
  using Tensor2D = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  // 0 1 2      2 1 0
  // 3 4 5   -> 5 4 3
  // 6 7 8      8 7 6
  Tensor2D tensor_2d;
  tensor_2d.setValues(
    {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}, {8, 18}, {9, 19}});

  Tensor2D expected_2d;
  expected_2d.setValues(
    {{2, 12}, {1, 11}, {0, 10}, {5, 15}, {4, 14}, {3, 13}, {8, 18}, {7, 17}, {6, 16}, {9, 19}});

  eigen_util::mirror_horizontal<3>(tensor_2d);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(tensor_2d(i, j), expected_2d(i, j)) << "tensor_2d:\n"
                                                    << tensor_2d << "\nexpected_2d:\n"
                                                    << expected_2d;
    }
  }

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M>>;
  Tensor tensor;
  tensor.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor expected;
  expected.setValues({2, 1, 0, 5, 4, 3, 8, 7, 6, 9});
  eigen_util::mirror_horizontal<3>(tensor);

  for (int i = 0; i < M; ++i) {
    EXPECT_EQ(tensor(i), expected(i)) << "tensor:\n" << tensor << "\nexpected:\n" << expected;
  }
}

TEST(eigen_util, flip_main_diag) {
  constexpr int M = 10;
  constexpr int N = 2;
  using Tensor2D = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  // 0 1 2      0 3 6
  // 3 4 5   -> 1 4 7
  // 6 7 8      2 5 8
  Tensor2D tensor_2d;
  tensor_2d.setValues(
    {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}, {8, 18}, {9, 19}});

  Tensor2D expected_2d;
  expected_2d.setValues(
    {{0, 10}, {3, 13}, {6, 16}, {1, 11}, {4, 14}, {7, 17}, {2, 12}, {5, 15}, {8, 18}, {9, 19}});

  eigen_util::flip_main_diag<3>(tensor_2d);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(tensor_2d(i, j), expected_2d(i, j)) << "tensor_2d:\n"
                                                    << tensor_2d << "\nexpected_2d:\n"
                                                    << expected_2d;
    }
  }

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M>>;
  Tensor tensor;
  tensor.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor expected;
  expected.setValues({0, 3, 6, 1, 4, 7, 2, 5, 8, 9});
  eigen_util::flip_main_diag<3>(tensor);

  for (int i = 0; i < M; ++i) {
    EXPECT_EQ(tensor(i), expected(i)) << "tensor:\n" << tensor << "\nexpected:\n" << expected;
  }
}

TEST(eigen_util, flip_anti_diag) {
  constexpr int M = 10;
  constexpr int N = 2;
  using Tensor2D = eigen_util::FTensor<Eigen::Sizes<M, N>>;

  // 0 1 2      8 5 2
  // 3 4 5   -> 7 4 1
  // 6 7 8      6 3 0
  Tensor2D tensor_2d;
  tensor_2d.setValues(
    {{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}, {8, 18}, {9, 19}});

  Tensor2D expected_2d;
  expected_2d.setValues(
    {{8, 18}, {5, 15}, {2, 12}, {7, 17}, {4, 14}, {1, 11}, {6, 16}, {3, 13}, {0, 10}, {9, 19}});

  eigen_util::flip_anti_diag<3>(tensor_2d);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(tensor_2d(i, j), expected_2d(i, j)) << "tensor_2d:\n"
                                                    << tensor_2d << "\nexpected_2d:\n"
                                                    << expected_2d;
    }
  }

  using Tensor = eigen_util::FTensor<Eigen::Sizes<M>>;
  Tensor tensor;
  tensor.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor expected;
  expected.setValues({8, 5, 2, 7, 4, 1, 6, 3, 0, 9});
  eigen_util::flip_anti_diag<3>(tensor);

  for (int i = 0; i < M; ++i) {
    EXPECT_EQ(tensor(i), expected(i)) << "tensor:\n" << tensor << "\nexpected:\n" << expected;
  }
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

  // --- 2D CASE ---
  {
    using Tensor = eigen_util::FTensor<Eigen::Sizes<2, 4>>;
    Tensor tensor;

    // Fill with row-major increasing pattern
    int counter = 0;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 4; ++j) tensor(i, j) = counter++;

    // Rotate last dim (columns) left by 1
    eigen_util::left_rotate(tensor, 1);
    // Expected result:
    // [[1,2,3,0],
    //  [5,6,7,4]]
    Tensor expected;
    expected.setValues({{1, 2, 3, 0}, {5, 6, 7, 4}});
    EXPECT_TRUE(eigen_util::equal(tensor, expected));

    // Rotate right by 1 (restore)
    eigen_util::right_rotate(tensor, 1);
    counter = 0;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 4; ++j) EXPECT_EQ(tensor(i, j), counter++);
  }

  // --- 3D CASE ---
  {
    constexpr int I = 2, J = 2, K = 4;
    using Tensor = eigen_util::FTensor<Eigen::Sizes<I, J, K>>;

    auto fill_base = [](Tensor& t) {
      for (int i = 0; i < I; ++i)
        for (int j = 0; j < J; ++j)
          for (int k = 0; k < K; ++k)
            // Encode (i,j,k) so each last-dim slice is [..,0],[..,1],[..,2],[..,3]
            t(i, j, k) = 100 * i + 10 * j + k;
    };

    auto expect_left = [&](const Tensor& base, int n, const Tensor& got) {
      n = ((n % K) + K) % K;  // normalize
      for (int i = 0; i < I; ++i)
        for (int j = 0; j < J; ++j)
          for (int k = 0; k < K; ++k) {
            // left-rotate by n: new[k] = old[(k+n) % K]
            int want = base(i, j, (k + n) % K);
            EXPECT_EQ(got(i, j, k), want)
              << "left n=" << n << " at (" << i << "," << j << "," << k << ")";
          }
    };

    auto expect_right = [&](const Tensor& base, int n, const Tensor& got) {
      n = ((n % K) + K) % K;  // normalize
      for (int i = 0; i < I; ++i)
        for (int j = 0; j < J; ++j)
          for (int k = 0; k < K; ++k) {
            // right-rotate by n: new[k] = old[(k - n + K) % K]
            int want = base(i, j, (k - n + K) % K);
            EXPECT_EQ(got(i, j, k), want)
              << "right n=" << n << " at (" << i << "," << j << "," << k << ")";
          }
    };

    // Base tensor
    Tensor base;
    fill_base(base);

    // 1) Left by 1 really is left by 1
    {
      Tensor t = base;
      eigen_util::left_rotate(t, 1);
      expect_left(base, 1, t);
    }

    // 2) Right by 1 really is right by 1
    {
      Tensor t = base;
      eigen_util::right_rotate(t, 1);
      expect_right(base, 1, t);
    }

    // 3) Left by 1 then right by 1 restores base
    {
      Tensor t = base;
      eigen_util::left_rotate(t, 1);
      eigen_util::right_rotate(t, 1);
      EXPECT_TRUE(eigen_util::equal(t, base));
    }

    // 4) Equivalence: left by 3 == right by 1 (since K==4)
    {
      Tensor tL = base, tR = base;
      eigen_util::left_rotate(tL, 3);
      eigen_util::right_rotate(tR, 1);
      EXPECT_TRUE(eigen_util::equal(tL, tR));
    }

    // 5) Mod behavior: left by 5 == left by 1
    {
      Tensor t1 = base, t5 = base;
      eigen_util::left_rotate(t1, 1);
      eigen_util::left_rotate(t5, 5);
      EXPECT_TRUE(eigen_util::equal(t1, t5));
    }

    // 6) Composition: left by 2 twice == identity (since 2+2 == 4 ≡ 0 mod K)
    {
      Tensor t = base;
      eigen_util::left_rotate(t, 2);
      eigen_util::left_rotate(t, 2);
      EXPECT_TRUE(eigen_util::equal(t, base));
    }
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
    {"ansi", [](float x, int) { return "\033[32m\u25CF\033[00m"; }},  // green circle
    {"col1", [](float x, int) { return "foo" + std::to_string((int)x); }},
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
  using Array1f = Eigen::Array<float, 4, 1>;
  using Array1i = Eigen::Array<int, 4, 1>;
  using Array2 = Eigen::Array<float, 4, 3>;

  Array1f a{1, 2, 3, 4};
  Array1i b{5, 6, 7, 8};
  Array1f c{9, 10, 11, 12};

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

TEST(EigenUtil, output_to_json_no_fmt) {
  Eigen::ArrayXXf a(2, 3);
  a << 1.0f, 2.0f, -3.25f, 4.0f, 5.5f, 6.0f;
  std::vector<std::string> keys{"a", "b", "c"};

  boost::json::object obj = eigen_util::output_to_json(a, keys, nullptr);

  const std::string output = boost::json::serialize(obj);
  const std::string expected = R"({"a":[1E0,4E0],"b":[2E0,5.5E0],"c":[-3.25E0,6E0]})";

  EXPECT_EQ(output, expected);
}

TEST(EigenUtil, output_to_json_with_fmt) {
  Eigen::ArrayXXf a(2, 3);
  a << 1.0f, 2.0f, -3.25f, 4.0f, 5.5f, 6.0f;
  std::vector<std::string> keys{"a", "b", "c"};

  eigen_util::PrintArrayFormatMap fmt;
  fmt["b"] = [](float x, int) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << "v=" << x;
    return os.str();
  };

  boost::json::object obj = eigen_util::output_to_json(a, keys, &fmt);

  const std::string output = boost::json::serialize(obj);
  const std::string expected = R"({"a":[1E0,4E0],"b":["v=2.00","v=5.50"],"c":[-3.25E0,6E0]})";

  EXPECT_EQ(output, expected);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
