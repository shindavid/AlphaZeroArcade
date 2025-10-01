#pragma once

#include "games/othello/BasicTypes.hpp"

namespace othello {

namespace detail {

constexpr int idx(int f, int r) { return r * 8 + f; }
constexpr bool in_bounds(int f, int r) { return (unsigned)f < 8 && (unsigned)r < 8; }
constexpr mask_t bit(int f, int r) { return mask_t{1} << idx(f, r); }
constexpr uint8_t bit8(int x) { return uint8_t(1u << x); }

}  // namespace detail
}
