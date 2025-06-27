#pragma once

#include <games/hex/Constants.hpp>
#include <games/hex/Types.hpp>

namespace hex {

// Utility class to efficiently reverse an 11-bit mask. Used for symmetry operations in Hex.
struct MaskReverser {
  static constexpr int B = Constants::kBoardDim;

  // Build a compile-time table of size 2^11 = 2048
  static constexpr std::array<mask_t, 1 << B> kLookUp = []() {
    std::array<mask_t, 1 << B> tbl = {};
    for (int x = 0; x < (1 << B); ++x) {
      // reverse the B-bit number x into y
      int y = 0;
      for (int i = 0; i < B; ++i)
        if (x & (1 << i)) y |= (1 << (B - 1 - i));
      tbl[x] = mask_t(y);
    }
    return tbl;
  }();

  static mask_t reverse(mask_t x) { return kLookUp[x]; }
};

}  // namespace hex
