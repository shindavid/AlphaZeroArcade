#include "util/FiniteGroups.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

// Verify group axioms for any FiniteGroup.
template <group::concepts::FiniteGroup G>
void verify_group_axioms() {
  constexpr int N = G::kOrder;

  for (int x = 0; x < N; ++x) {
    // Identity: compose(x, e) == x and compose(e, x) == x
    EXPECT_EQ(G::compose(x, group::kIdentity), x) << "Right identity failed for x=" << x;
    EXPECT_EQ(G::compose(group::kIdentity, x), x) << "Left identity failed for x=" << x;

    // Inverse: compose(x, inv(x)) == e and compose(inv(x), x) == e
    auto inv = G::inverse(x);
    EXPECT_EQ(G::compose(x, inv), group::kIdentity)
      << "Right inverse failed for x=" << x << " inv=" << inv;
    EXPECT_EQ(G::compose(inv, x), group::kIdentity)
      << "Left inverse failed for x=" << x << " inv=" << inv;

    // Closure: compose(x, y) is in [0, N)
    for (int y = 0; y < N; ++y) {
      auto z = G::compose(x, y);
      EXPECT_GE(z, 0) << "Closure failed for x=" << x << " y=" << y;
      EXPECT_LT(z, N) << "Closure failed for x=" << x << " y=" << y;
    }
  }

  // Associativity: compose(x, compose(y, z)) == compose(compose(x, y), z)
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      for (int z = 0; z < N; ++z) {
        auto lhs = G::compose(x, G::compose(y, z));
        auto rhs = G::compose(G::compose(x, y), z);
        EXPECT_EQ(lhs, rhs) << "Associativity failed for x=" << x << " y=" << y << " z=" << z;
      }
    }
  }
}

TEST(FiniteGroups, TrivialGroupAxioms) { verify_group_axioms<groups::TrivialGroup>(); }

TEST(FiniteGroups, C2Axioms) { verify_group_axioms<groups::C2>(); }

TEST(FiniteGroups, CyclicGroup3Axioms) { verify_group_axioms<groups::CyclicGroup<3>>(); }

TEST(FiniteGroups, CyclicGroup5Axioms) { verify_group_axioms<groups::CyclicGroup<5>>(); }

TEST(FiniteGroups, D1Axioms) { verify_group_axioms<groups::D1>(); }

TEST(FiniteGroups, D4Axioms) { verify_group_axioms<groups::D4>(); }

TEST(FiniteGroups, DihedralGroup3Axioms) { verify_group_axioms<groups::DihedralGroup<3>>(); }

TEST(FiniteGroups, D4NamedElements) {
  // Verify specific composition identities for D4
  using D = groups::D4;

  // rot90 composed 4 times = identity
  auto x = D::kRot90;
  x = D::compose(x, D::kRot90);
  EXPECT_EQ(x, D::kRot180);
  x = D::compose(x, D::kRot90);
  EXPECT_EQ(x, D::kRot270);
  x = D::compose(x, D::kRot90);
  EXPECT_EQ(x, D::kIdentity);

  // Reflections are self-inverse
  EXPECT_EQ(D::inverse(D::kFlipVertical), D::kFlipVertical);
  EXPECT_EQ(D::inverse(D::kFlipMainDiag), D::kFlipMainDiag);
  EXPECT_EQ(D::inverse(D::kMirrorHorizontal), D::kMirrorHorizontal);
  EXPECT_EQ(D::inverse(D::kFlipAntiDiag), D::kFlipAntiDiag);

  // rot90 and rot270 are inverses
  EXPECT_EQ(D::inverse(D::kRot90), D::kRot270);
  EXPECT_EQ(D::inverse(D::kRot270), D::kRot90);

  // rot180 is self-inverse
  EXPECT_EQ(D::inverse(D::kRot180), D::kRot180);
}

TEST(FiniteGroups, InverseOfIdentityIsIdentity) {
  EXPECT_EQ(groups::TrivialGroup::inverse(0), 0);
  EXPECT_EQ(groups::C2::inverse(0), 0);
  EXPECT_EQ(groups::D4::inverse(0), 0);
  EXPECT_EQ(groups::CyclicGroup<5>::inverse(0), 0);
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
