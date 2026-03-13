#include "util/GTestUtil.hpp"
#include "util/Math.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#ifndef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must be defined for unit tests");
#endif

namespace {

// Reference inverse normal CDF via bisection.
// Slow but robust for unit testing.
inline float ref_inv_normal_cdf(float p) {
  p = std::clamp(p, 1e-8f, 1.0f - 1e-8f);

  float lo = -10.0f;
  float hi = 10.0f;

  for (int it = 0; it < 80; ++it) {
    float mid = 0.5f * (lo + hi);
    float cmid = math::normal_cdf(mid);
    if (cmid < p)
      lo = mid;
    else
      hi = mid;
  }
  return 0.5f * (lo + hi);
}

// Expected output under the box-eps model.
inline float expected_clamped_z(float p0, float pi, float eps) {
  const float denom = p0 + pi;
  if (denom <= 0.0f) return 0.0f;

  const float num_min = std::max(0.0f, p0 - eps);
  const float num_max = p0 + eps;

  const float rmin = num_min / denom;
  const float rmax = num_max / denom;

  const float y_min = ref_inv_normal_cdf(rmin);
  const float y_max = ref_inv_normal_cdf(rmax);

  return std::clamp(0.0f, y_min, y_max);
}

// Exact reference logit (with safe clamp for testing)
inline float ref_logit(float mu) {
  const float eps = 1e-8f;
  mu = std::min(1.0f - eps, std::max(eps, mu));
  return std::log(mu / (1.0f - mu));
}

inline float ref_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---------- Long-double stable reference for log(alpha) ----------

static inline long double logPhiNeg_moderate_ld(long double z_nonneg) {
  static const long double SQRT1_2_LD = 1.0l / std::sqrt(2.0l);
  const long double t = z_nonneg * SQRT1_2_LD;
  return std::log(0.5l) + std::log(std::erfc(t));
}

static inline long double logPhi_moderate_ld(long double z) {
  static const long double SQRT1_2_LD = 1.0l / std::sqrt(2.0l);
  if (z < 0.0l) {
    const long double t = -z * SQRT1_2_LD;
    return std::log(0.5l) + std::log(std::erfc(t));
  } else {
    const long double s = logPhiNeg_moderate_ld(z);
    return std::log1p(-std::exp(s));
  }
}

static inline long double log_odds_normal_ld(long double z) {
  static const long double LOG_SQRT_2PI_LD = 0.5l * std::log(2.0l * M_PIl);
  static const long double THRESH = 10.0l;

  if (z < -THRESH) {
    const long double t = -z;
    const long double inv = 1.0l / t;
    const long double inv3 = inv * inv * inv;
    const long double inv5 = inv3 * inv * inv;
    const long double inv7 = inv5 * inv * inv;
    const long double R = inv - inv3 + 3.0l * inv5 - 15.0l * inv7;
    const long double logphi = -0.5l * t * t - LOG_SQRT_2PI_LD;
    return logphi + std::log(R);
  }
  if (z > THRESH) {
    return -log_odds_normal_ld(-z);
  }
  return logPhi_moderate_ld(z) - logPhi_moderate_ld(-z);
}

static inline long double log_alpha_ref_ld(long double z_new, long double z_old) {
  return log_odds_normal_ld(z_new) - log_odds_normal_ld(z_old);
}

static inline ::testing::AssertionResult NearAbsLog(double log_a, long double log_b,
                                                    double tol_abs) {
  const long double diff = std::fabs((long double)log_a - log_b);
  if (diff <= tol_abs) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "log_a=" << log_a << " log_b=" << (double)log_b
                                       << " |diff|=" << (double)diff << " > tol=" << tol_abs;
}

}  // namespace

TEST(math, fast_coarse_log_less_than_1_basic_half) {
  const float exact = std::log(0.5f);
  EXPECT_NEAR(math::fast_coarse_log_less_than_1(0.5f), exact, 2e-3f);
}

TEST(math, fast_coarse_log_less_than_1_sign_sanity) {
  const float xs[] = {0.999f, 0.9f, 0.75f, 0.5f, 0.25f, 0.1f, 0.01f};

  for (float x : xs) {
    float v = math::fast_coarse_log_less_than_1(x);
    EXPECT_LT(v, 0.0f) << "x=" << x;
  }
}

TEST(math, fast_coarse_log_less_than_1_monotonic_on_dense_grid) {
  float prev = math::fast_coarse_log_less_than_1(0.001f);

  for (int k = 2; k <= 999; ++k) {
    float x = 0.001f * k;
    float cur = math::fast_coarse_log_less_than_1(x);

    EXPECT_LE(prev, cur + 1e-4f) << "x=" << x;
    prev = cur;
  }
}

TEST(math, fast_coarse_log_less_than_1_matches_std_log_on_0p01_grid) {
  const float tol = 0.02f;

  for (int k = 1; k <= 99; ++k) {
    float x = 0.01f * k;
    float fast = math::fast_coarse_log_less_than_1(x);
    float exact = std::log(x);

    EXPECT_NEAR(fast, exact, tol) << "x=" << x;
  }
}

TEST(math, fast_coarse_log_less_than_1_edge_behavior_is_finite_and_ordered) {
  const float xs[] = {1e-6f, 1e-4f, 1e-3f, 1e-2f, 0.1f, 0.5f, 0.9f, 0.99f, 0.999f};

  float prev = math::fast_coarse_log_less_than_1(xs[0]);
  for (int i = 1; i < static_cast<int>(sizeof(xs) / sizeof(xs[0])); ++i) {
    float cur = math::fast_coarse_log_less_than_1(xs[i]);

    EXPECT_LE(prev, cur + 1e-3f) << "x_prev=" << xs[i - 1] << " x_cur=" << xs[i];
    prev = cur;
  }

  float near1 = math::fast_coarse_log_less_than_1(0.999999f);
  EXPECT_LT(near1, 0.0f);
  EXPECT_GT(near1, -1e-2f);
}

TEST(math, fast_coarse_logit_basic_midpoint) {
  EXPECT_NEAR(math::fast_coarse_logit(0.5f), 0.0f, 1e-5f);
}

TEST(math, fast_coarse_logit_sign_sanity) {
  EXPECT_LT(math::fast_coarse_logit(0.25f), 0.0f);
  EXPECT_LT(math::fast_coarse_logit(0.10f), 0.0f);

  EXPECT_GT(math::fast_coarse_logit(0.75f), 0.0f);
  EXPECT_GT(math::fast_coarse_logit(0.90f), 0.0f);
}

TEST(math, fast_coarse_logit_odd_symmetry_about_half) {
  const float tol = 1e-5f;

  for (int k = 1; k <= 40; ++k) {
    float d = 0.01f * k;
    float a = 0.5f - d;
    float b = 0.5f + d;

    float fa = math::fast_coarse_logit(a);
    float fb = math::fast_coarse_logit(b);

    EXPECT_NEAR(fb, -fa, tol) << "d=" << d;
  }
}

TEST(math, fast_coarse_logit_monotonic_on_dense_grid) {
  float prev = math::fast_coarse_logit(0.001f);

  for (int k = 1; k <= 999; ++k) {
    float mu = 0.001f * k;
    float cur = math::fast_coarse_logit(mu);

    EXPECT_LE(prev, cur + 1e-4f) << "mu=" << mu;

    prev = cur;
  }
}

TEST(math, fast_coarse_logit_matches_logit_on_0p01_grid) {
  const float tol = .02f;

  for (int k = 1; k <= 99; ++k) {
    float mu = 0.01f * k;

    float fast = math::fast_coarse_logit(mu);
    float exact = ref_logit(mu);

    EXPECT_NEAR(fast, exact, tol) << "mu=" << mu;
  }
}

TEST(math, fast_coarse_logit_edge_behavior_is_finite_and_signed) {
  const float mus[] = {1e-6f, 1e-4f, 1e-3f, 1.0f - 1e-3f, 1.0f - 1e-4f, 1.0f - 1e-6f};

  for (float mu : mus) {
    float v = math::fast_coarse_logit(mu);

    if (mu < 0.5f) {
      EXPECT_LE(v, 0.0f) << "mu=" << mu;
    } else if (mu > 0.5f) {
      EXPECT_GE(v, 0.0f) << "mu=" << mu;
    }
  }
}

TEST(math, fast_coarse_sigmoid_matches_reference_on_grid) {
  constexpr float kTol = 1e-4f;

  for (int k = -800; k <= 800; ++k) {
    float x = 0.01f * static_cast<float>(k);

    float ref = ref_sigmoid(x);
    float fast = math::fast_coarse_sigmoid(x);

    EXPECT_NEAR(fast, ref, kTol) << "x=" << x;
  }
}

TEST(math, fast_coarse_sigmoid_saturates_in_tails) {
  float y_neg = math::fast_coarse_sigmoid(-100.0f);
  float y_pos = math::fast_coarse_sigmoid(100.0f);

  EXPECT_NEAR(y_neg, 0.0f, 1e-5f);
  EXPECT_NEAR(y_pos, 1.0f, 1e-5f);
}

TEST(math, fast_coarse_sigmoid_is_monotone_non_decreasing) {
  float prev = math::fast_coarse_sigmoid(-8.0f);

  for (int i = 1; i <= 400; ++i) {
    float x = -8.0f + 16.0f * (static_cast<float>(i) / 400.0f);
    float y = math::fast_coarse_sigmoid(x);

    EXPECT_GE(y + 1e-6f, prev) << "Monotonicity violated at x=" << x;
    prev = y;
  }
}

TEST(math, fast_coarse_batch_normal_cdf) {
  constexpr int kMin = -200;
  constexpr int kMax = 200;
  constexpr int n = (kMax - kMin + 1);

  std::vector<float> x;
  x.reserve(n);
  for (int k = kMin; k <= kMax; ++k) {
    x.push_back(0.01f * static_cast<float>(k));
  }

  std::vector<float> y(x.size(), -1.0f);

  math::fast_coarse_batch_normal_cdf(x.data(), static_cast<int>(x.size()), y.data());

  for (size_t i = 0; i < y.size(); ++i) {
    EXPECT_GE(y[i], 0.0f) << "y[" << i << "]";
    EXPECT_LE(y[i], 1.0f) << "y[" << i << "]";
  }

  for (size_t i = 1; i < y.size(); ++i) {
    EXPECT_LE(y[i - 1], y[i]) << "Non-monotone at i=" << i << " x[i-1]=" << x[i - 1]
                              << " x[i]=" << x[i] << " y[i-1]=" << y[i - 1] << " y[i]=" << y[i];
  }

  const float tol = 1e-4f;

  for (size_t i = 0; i < x.size(); ++i) {
    float exact = math::normal_cdf(x[i]);
    EXPECT_NEAR(y[i], exact, tol) << "x=" << x[i];
  }

  const size_t mid = static_cast<size_t>(-kMin);
  ASSERT_LT(mid, x.size());
  EXPECT_NEAR(x[mid], 0.0f, 1e-8f);
  EXPECT_NEAR(y[mid], 0.5f, 0.01f);
}

TEST(math, fast_coarse_batch_normal_cdf_repeated_values) {
  float values[] = {-17.0f, -1.0f, 0.0f, 0.1f, 3.0f, +17.0f};
  for (float v : values) {
    std::vector<float> x(1024, v);
    std::vector<float> y(1024, -1.0f);

    math::fast_coarse_batch_normal_cdf(x.data(), static_cast<int>(x.size()), y.data());

    EXPECT_GE(y[0], 0.0f) << "v=" << v;
    EXPECT_LE(y[0], 1.0f) << "v=" << v;
    for (size_t i = 1; i < y.size(); ++i) {
      EXPECT_EQ(y[i], y[0]) << "v=" << v << " i=" << i;
    }
  }
}

TEST(math, fast_coarse_batch_normal_cdf_edge_cases) {
  {
    std::vector<float> x;
    std::vector<float> y;
    math::fast_coarse_batch_normal_cdf(x.data(), 0, y.data());
    SUCCEED();
  }

  {
    constexpr int n = 5;
    float x[n] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float y[n] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    math::fast_coarse_batch_normal_cdf(x, n, y);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(y[i], 0.5f) << "x=" << x[i];
    }
  }

  {
    constexpr int n = 3;
    float x[n] = {-20.0f, -50.0f, -100.0f};
    float y[n] = {-1.0f, -1.0f, -1.0f};
    math::fast_coarse_batch_normal_cdf(x, n, y);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(y[i], 0.0f) << "x=" << x[i];
    }
  }

  {
    constexpr int n = 3;
    float x[n] = {20.0f, 50.0f, 100.0f};
    float y[n] = {-1.0f, -1.0f, -1.0f};
    math::fast_coarse_batch_normal_cdf(x, n, y);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(y[i], 1.0f) << "x=" << x[i];
    }
  }
}

TEST(math, fast_coarse_batch_inverse_normal_cdf_clamped_range_n0) {
  float p0 = 0.2f;
  const float eps = 0.01f;

  std::vector<float> p;
  std::vector<float> c;
  std::vector<float> y;
  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(p0, p.data(), c.data(), p.size(),
                                                           y.data(), eps);
  SUCCEED();
}

TEST(math, fast_coarse_batch_inverse_normal_cdf_clamped_range_equal_probs_gives_zero) {
  const float p0 = 0.2f;
  const float eps = 0.01f;

  std::vector<float> p = {0.2f, 0.2f, 0.2f, 0.2f};
  std::vector<float> c(p.size(), 0.0f);
  std::vector<float> y(p.size(), 123.0f);

  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(p0, p.data(), c.data(), p.size(),
                                                           y.data(), eps);

  for (size_t i = 0; i < y.size(); ++i) {
    EXPECT_NEAR(y[i], 0.0f, 1e-3f) << "i=" << i;
  }
}

TEST(math, fast_coarse_batch_inverse_normal_cdf_clamped_range_sign_sanity) {
  const float eps = 0.01f;

  {
    float p0 = 0.30f;
    std::vector<float> p = {0.05f, 0.10f, 0.15f};
    std::vector<float> c(p.size(), 0.0f);
    std::vector<float> y(p.size(), 0.0f);

    math::fast_coarse_batch_inverse_normal_cdf_clamped_range(p0, p.data(), c.data(), p.size(),
                                                             y.data(), eps);

    for (float v : y) {
      EXPECT_GT(v, 0.0f);
    }
  }

  {
    float p0 = 0.10f;
    std::vector<float> p = {0.20f, 0.30f, 0.50f};
    std::vector<float> c(p.size(), 0.0f);
    std::vector<float> y(p.size(), 0.0f);

    math::fast_coarse_batch_inverse_normal_cdf_clamped_range(p0, p.data(), c.data(), p.size(),
                                                             y.data(), eps);

    for (float v : y) {
      EXPECT_LT(v, 0.0f);
    }
  }
}

TEST(math, fast_coarse_batch_inverse_normal_cdf_clamped_range_eps_zero_matches_point_ratio) {
  const float p0 = 0.30f;
  const float pi = 0.10f;
  const float eps = 0.0f;

  std::vector<float> p = {pi};
  std::vector<float> c(p.size(), 0.0f);
  std::vector<float> y(p.size(), 0.0f);

  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(p0, p.data(), c.data(), p.size(),
                                                           y.data(), eps);

  const float r = p0 / (p0 + pi);
  const float z_ref = ref_inv_normal_cdf(r);

  EXPECT_NEAR(y[0], z_ref, 1e-2f);
}

TEST(math, fast_coarse_batch_inverse_normal_cdf_clamped_range_matches_reference_model_dense_grid) {
  const float p0 = 0.23f;
  const float eps = 0.01f;

  std::vector<float> p;
  p.reserve(101);
  for (int k = 0; k <= 100; ++k) {
    p.push_back(0.01f * static_cast<float>(k));
  }

  std::vector<float> c(p.size(), 0.0f);
  std::vector<float> y(p.size(), 0.0f);

  math::fast_coarse_batch_inverse_normal_cdf_clamped_range(p0, p.data(), c.data(), p.size(),
                                                           y.data(), eps);

  for (size_t i = 0; i < p.size(); ++i) {
    float expected = expected_clamped_z(p0, p[i], eps);

    constexpr float tol = .0001f;

    EXPECT_NEAR(y[i], expected, tol)
      << "i=" << i << " p[i]=" << p[i] << " p0=" << p0 << " eps=" << eps;
  }
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

TEST(math, normal_cdf_logit_diff_IdentityWhenEqual) {
  EXPECT_DOUBLE_EQ(math::normal_cdf_logit_diff(-3.14, -3.14), 0.0);
  EXPECT_DOUBLE_EQ(math::normal_cdf_logit_diff(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(math::normal_cdf_logit_diff(25.0, 25.0), 0.0);
}

TEST(math, normal_cdf_logit_diff_Antisymmetry) {
  std::vector<std::pair<double, double>> cases = {
    {-2.0, -1.0}, {-6.0, -5.5}, {1.0, 2.5}, {-9.0, 3.0}, {7.5, -7.2}};
  for (auto [z1, z2] : cases) {
    const double d12 = math::normal_cdf_logit_diff(z1, z2);
    const double d21 = math::normal_cdf_logit_diff(z2, z1);
    EXPECT_NEAR(d12 + d21, 0.0, 1e-12);
  }
}

TEST(math, normal_cdf_logit_diff_Monotonicity) {
  EXPECT_GT(math::normal_cdf_logit_diff(-1.0, -2.0), 0.0);
  EXPECT_GT(math::normal_cdf_logit_diff(2.0, 1.0), 0.0);
  EXPECT_LT(math::normal_cdf_logit_diff(-2.0, -1.0), 0.0);
}

TEST(math, normal_cdf_logit_diff_MatchesNaiveInModerateRange) {
  auto naive_log_diff = [](double zn, double zo) {
    const double p_new = 0.5 * std::erfc(-zn / std::sqrt(2.0));
    const double p_old = 0.5 * std::erfc(-zo / std::sqrt(2.0));
    const double lnew = std::log(p_new) - std::log1p(-p_new);
    const double lold = std::log(p_old) - std::log1p(-p_old);
    return lnew - lold;
  };

  std::vector<std::pair<double, double>> cases = {
    {-2.0, -1.0}, {-1.5, -1.6}, {0.0, -0.5}, {2.0, 1.0}, {3.0, -3.0}};

  for (auto [zn, zo] : cases) {
    const double stable_log = math::normal_cdf_logit_diff(zn, zo);
    const double naive_log = naive_log_diff(zn, zo);
    EXPECT_NEAR(stable_log, naive_log, 1e-12);
  }
}

TEST(math, normal_cdf_logit_diff_ExtremeNegative_AgreesWithLongDouble) {
  std::vector<std::pair<double, double>> cases = {{-20.0, -19.5}, {-30.0, -29.0}, {-39.0, -38.5}};

  for (auto [zn, zo] : cases) {
    const double stable_log = math::normal_cdf_logit_diff(zn, zo);
    const long double ref_log = log_alpha_ref_ld((long double)zn, (long double)zo);
    EXPECT_TRUE(NearAbsLog(stable_log, ref_log, /*tol_abs=*/1e-9));
  }
}

TEST(math, normal_cdf_logit_diff_ExtremePositive_AgreesWithLongDouble) {
  std::vector<std::pair<double, double>> cases = {{19.5, 20.0}, {29.0, 30.0}, {38.5, 39.0}};

  for (auto [zn, zo] : cases) {
    const double stable_log = math::normal_cdf_logit_diff(zn, zo);
    const long double ref_log = log_alpha_ref_ld((long double)zn, (long double)zo);
    EXPECT_TRUE(NearAbsLog(stable_log, ref_log, /*tol_abs=*/1e-9));
  }
}

TEST(math, normal_cdf_logit_diff_VectorizedSweepSanity) {
  std::vector<double> zs = {-10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0};
  for (double zo : zs) {
    for (double zn : zs) {
      const double d = math::normal_cdf_logit_diff(zn, zo);
      if (zn > zo) {
        EXPECT_GT(d, 0.0);
      } else if (zn < zo) {
        EXPECT_LT(d, 0.0);
      }
      const long double ref_log = log_alpha_ref_ld((long double)zn, (long double)zo);
      EXPECT_TRUE(NearAbsLog(d, ref_log, 1e-5));
    }
  }
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
