#include "util/BitMapUtil.hpp"
#include "util/GTestUtil.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

// Helper: set bit at row r, col c in an 8x8 bitmap (row 0 = top = MSByte)
constexpr uint64_t bit(int r, int c) { return 1ULL << (r * 8 + c); }

}  // namespace

TEST(BitMapUtil, FlipVertical) {
  // Place a single bit at (0, 0). After flip_vertical, it should be at (7, 0).
  uint64_t m = bit(0, 0);
  bitmap_util::flip_vertical(m);
  EXPECT_EQ(m, bit(7, 0));

  // Bit at (1, 3) -> (6, 3)
  m = bit(1, 3);
  bitmap_util::flip_vertical(m);
  EXPECT_EQ(m, bit(6, 3));
}

TEST(BitMapUtil, MirrorHorizontal) {
  // Bit at (0, 0) -> (0, 7)
  uint64_t m = bit(0, 0);
  bitmap_util::mirror_horizontal(m);
  EXPECT_EQ(m, bit(0, 7));

  // Bit at (3, 1) -> (3, 6)
  m = bit(3, 1);
  bitmap_util::mirror_horizontal(m);
  EXPECT_EQ(m, bit(3, 6));
}

TEST(BitMapUtil, FlipMainDiag) {
  // Bit at (r, c) -> (c, r)
  uint64_t m = bit(0, 3);
  bitmap_util::flip_main_diag(m);
  EXPECT_EQ(m, bit(3, 0));

  m = bit(2, 5);
  bitmap_util::flip_main_diag(m);
  EXPECT_EQ(m, bit(5, 2));
}

TEST(BitMapUtil, FlipAntiDiag) {
  // Bit at (r, c) -> (7-c, 7-r)
  uint64_t m = bit(0, 0);
  bitmap_util::flip_anti_diag(m);
  EXPECT_EQ(m, bit(7, 7));

  m = bit(1, 2);
  bitmap_util::flip_anti_diag(m);
  EXPECT_EQ(m, bit(5, 6));
}

TEST(BitMapUtil, Rot90Clockwise) {
  // rot90_cw(r, c) -> (c, 7-r)
  uint64_t m = bit(0, 0);
  bitmap_util::rot90_clockwise(m);
  EXPECT_EQ(m, bit(0, 7));

  m = bit(1, 2);
  bitmap_util::rot90_clockwise(m);
  EXPECT_EQ(m, bit(2, 6));
}

TEST(BitMapUtil, Rot180) {
  // rot180(r, c) -> (7-r, 7-c)
  uint64_t m = bit(0, 0);
  bitmap_util::rot180(m);
  EXPECT_EQ(m, bit(7, 7));

  m = bit(1, 2);
  bitmap_util::rot180(m);
  EXPECT_EQ(m, bit(6, 5));
}

TEST(BitMapUtil, Rot270Clockwise) {
  // rot270_cw(r, c) -> (7-c, r)
  uint64_t m = bit(0, 0);
  bitmap_util::rot270_clockwise(m);
  EXPECT_EQ(m, bit(7, 0));

  m = bit(1, 2);
  bitmap_util::rot270_clockwise(m);
  EXPECT_EQ(m, bit(5, 1));
}

TEST(BitMapUtil, Rot90FourTimesIsIdentity) {
  uint64_t original = bit(1, 3) | bit(5, 2) | bit(7, 7);
  uint64_t m = original;
  bitmap_util::rot90_clockwise(m);
  bitmap_util::rot90_clockwise(m);
  bitmap_util::rot90_clockwise(m);
  bitmap_util::rot90_clockwise(m);
  EXPECT_EQ(m, original);
}

TEST(BitMapUtil, Rot180TwiceIsIdentity) {
  uint64_t original = bit(0, 1) | bit(3, 6) | bit(7, 4);
  uint64_t m = original;
  bitmap_util::rot180(m);
  bitmap_util::rot180(m);
  EXPECT_EQ(m, original);
}

TEST(BitMapUtil, FlipVerticalTwiceIsIdentity) {
  uint64_t original = bit(2, 3) | bit(5, 1);
  uint64_t m = original;
  bitmap_util::flip_vertical(m);
  bitmap_util::flip_vertical(m);
  EXPECT_EQ(m, original);
}

TEST(BitMapUtil, MirrorHorizontalTwiceIsIdentity) {
  uint64_t original = bit(4, 2) | bit(6, 7);
  uint64_t m = original;
  bitmap_util::mirror_horizontal(m);
  bitmap_util::mirror_horizontal(m);
  EXPECT_EQ(m, original);
}

TEST(BitMapUtil, FlipMainDiagTwiceIsIdentity) {
  uint64_t original = bit(1, 6) | bit(3, 0);
  uint64_t m = original;
  bitmap_util::flip_main_diag(m);
  bitmap_util::flip_main_diag(m);
  EXPECT_EQ(m, original);
}

TEST(BitMapUtil, FlipAntiDiagTwiceIsIdentity) {
  uint64_t original = bit(0, 4) | bit(7, 3);
  uint64_t m = original;
  bitmap_util::flip_anti_diag(m);
  bitmap_util::flip_anti_diag(m);
  EXPECT_EQ(m, original);
}

TEST(BitMapUtil, MultiMaskVariadicApply) {
  // Test that variadic versions work on multiple masks simultaneously
  uint64_t a = bit(0, 0);
  uint64_t b = bit(1, 1);
  bitmap_util::flip_vertical(a, b);
  EXPECT_EQ(a, bit(7, 0));
  EXPECT_EQ(b, bit(6, 1));
}

int main(int argc, char** argv) { return launch_gtest(argc, argv); }
