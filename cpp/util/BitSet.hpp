#pragma once

#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

/*
 * Helper facilities for std::bitset.
 */
namespace bitset_util {

/*
 * Usage:
 *
 * std::bitset<8> bits;
 * bits[1] = 1;
 *
 * for (int k : bitset_util::on_indices(bits)) {
 *   printf("set %d\n", k);
 * }
 * for (int k : bitset_util::off_indices(bits)) {
 *   printf("unset %d\n", k);
 * }
 *
 * Above prints:
 *
 * set 1
 * unset 0
 * unset 2
 * unset 3
 */
template<size_t N> auto on_indices(const std::bitset<N>&);
template<size_t N> auto off_indices(const std::bitset<N>&);

/*
 * Picks a uniform random on/off index and returns it.
 *
 * Assumes that at least one on/off index exists. Behavior is undefined if this assumption is violated.
 */
template<size_t N> int choose_random_on_index(const std::bitset<N>&);
template<size_t N> int choose_random_off_index(const std::bitset<N>&);

}  // namespace bitset_util

#include <util/inl/BitSet.inl>
