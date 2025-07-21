#pragma once

#include <cstdint>

/*
 * Various utility functions for manipulating bitmaps.
 *
 * Each function accepts one or more uint64_t references, each of which correspond to an 8x8 bitmap:
 *
 *  0  1  2  3  4  5  6  7
 *  8  9 10 11 12 13 14 15
 * 16 17 18 19 20 21 22 23
 * 24 25 26 27 28 29 30 31
 * 32 33 34 35 36 37 38 39
 * 40 41 42 43 44 45 46 47
 * 48 49 50 51 52 53 54 55
 * 56 57 58 59 60 61 62 63
 *
 * If the game representation uses a different layout, the functions won't act as advertised, but
 * in typical contexts that should be ok.
 *
 * In the function names, "main diagonal" refers to the diagonal from the top-left to the
 * bottom-right, and "anti-diagonal" refers to the diagonal from the top-right to the bottom-left.
 *
 * Implementations are taken from:
 *
 * https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating
 */
namespace bitmap_util {

template <typename... UInt64T> void flip_vertical(UInt64T&... mask);
template <typename... UInt64T> void mirror_horizontal(UInt64T&... mask);
template <typename... UInt64T> void flip_main_diag(UInt64T&... mask);
template <typename... UInt64T> void flip_anti_diag(UInt64T&... mask);
template <typename... UInt64T> void rot90_clockwise(UInt64T&... mask);
template <typename... UInt64T> void rot180(UInt64T&... mask);
template <typename... UInt64T> void rot270_clockwise(UInt64T&... mask);

}  // namespace bitmap_util

#include "inline/util/BitMapUtil.inl"
