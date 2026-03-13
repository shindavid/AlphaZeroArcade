#include "util/GTestUtil.hpp"
#include "util/Gaussian1D.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(Gaussian1D, DefaultIsInvalid) {
  util::Gaussian1D g;
  EXPECT_FALSE(g.valid());
}

TEST(Gaussian1D, ConstructAndAccess) {
  util::Gaussian1D g(1.5f, 0.25f);
  EXPECT_TRUE(g.valid());
  EXPECT_FLOAT_EQ(g.mean(), 1.5f);
  EXPECT_FLOAT_EQ(g.variance(), 0.25f);
}

TEST(Gaussian1D, NegInf) {
  auto g = util::Gaussian1D::neg_inf();
  EXPECT_TRUE(g.valid());
  EXPECT_FLOAT_EQ(g.variance(), util::Gaussian1D::kVarianceNegInf);
}

TEST(Gaussian1D, PosInf) {
  auto g = util::Gaussian1D::pos_inf();
  EXPECT_TRUE(g.valid());
  EXPECT_FLOAT_EQ(g.variance(), util::Gaussian1D::kVariancePosInf);
}

TEST(Gaussian1D, Equality) {
  util::Gaussian1D a(1.0f, 2.0f);
  util::Gaussian1D b(1.0f, 2.0f);
  util::Gaussian1D c(1.0f, 3.0f);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST(Gaussian1D, UnaryNegate) {
  // Normal case: negate flips mean, preserves variance
  util::Gaussian1D g(3.0f, 1.0f);
  auto ng = -g;
  EXPECT_FLOAT_EQ(ng.mean(), -3.0f);
  EXPECT_FLOAT_EQ(ng.variance(), 1.0f);
}

TEST(Gaussian1D, UnaryNegateInfSwaps) {
  // Negating neg_inf should give pos_inf and vice versa
  auto ni = util::Gaussian1D::neg_inf();
  auto pi = util::Gaussian1D::pos_inf();

  auto neg_ni = -ni;
  auto neg_pi = -pi;

  EXPECT_FLOAT_EQ(neg_ni.variance(), util::Gaussian1D::kVariancePosInf);
  EXPECT_FLOAT_EQ(neg_pi.variance(), util::Gaussian1D::kVarianceNegInf);
}

TEST(Gaussian1D, FmtMeanNormal) {
  std::string s = util::Gaussian1D::fmt_mean(1.5f, 0.25f);
  EXPECT_FALSE(s.empty());
}

TEST(Gaussian1D, FmtMeanSpecialValues) {
  EXPECT_EQ(util::Gaussian1D::fmt_mean(0, util::Gaussian1D::kVarianceNegInf), "-inf");
  EXPECT_EQ(util::Gaussian1D::fmt_mean(0, util::Gaussian1D::kVariancePosInf), "+inf");
  EXPECT_EQ(util::Gaussian1D::fmt_mean(0, util::Gaussian1D::kVarianceUnset), "???");
}

TEST(Gaussian1D, FmtVariance) {
  std::string s = util::Gaussian1D::fmt_variance(0.5f);
  EXPECT_FALSE(s.empty());

  // Negative variance (non-special) should be clamped to 0
  std::string s2 = util::Gaussian1D::fmt_variance(-0.1f);
  std::string s3 = util::Gaussian1D::fmt_variance(0.0f);
  EXPECT_EQ(s2, s3);  // -0.1 clamped to 0, same as passing 0 directly
}

TEST(Gaussian1D, FmtMean0SpecialValues) {
  EXPECT_EQ(util::Gaussian1D::fmt_mean0(0, util::Gaussian1D::kVarianceNegInf), "-inf");
  EXPECT_EQ(util::Gaussian1D::fmt_mean0(0, util::Gaussian1D::kVariancePosInf), "+inf");
  EXPECT_EQ(util::Gaussian1D::fmt_mean0(0, util::Gaussian1D::kVarianceUnset), "???");
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
