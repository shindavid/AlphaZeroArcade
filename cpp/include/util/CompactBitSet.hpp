#pragma once

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace util {

// util::CompactBitSet<N> is a mostly drop-in replacement for std::bitset<N> that uses a more
// compact storage representation for small N.
template <size_t N>
class CompactBitSet {
 public:
  using int_t = std::conditional_t<
    (N <= 8), uint8_t,
    std::conditional_t<(N <= 16), uint16_t, std::conditional_t<(N <= 32), uint32_t, uint64_t>>>;

 private:
  static_assert(N >= 1, "CompactBitSet requires N >= 1");

  static constexpr size_t B = sizeof(int_t) * 8;              // bits per word
  static constexpr size_t M = (N <= 32) ? 1 : (N + 63) / 64;  // num words
  using array_t = std::array<int_t, M>;

  array_t storage_{};

  static constexpr int_t all_ones() noexcept;
  static constexpr int_t tail_mask() noexcept;
  static constexpr void bounds_check(size_t pos);

  constexpr void mask_tail() noexcept;

 public:
  static constexpr size_t size() noexcept;
  bool operator[](size_t pos) const;
  bool test(size_t pos) const;
  bool any() const noexcept;
  bool none() const noexcept;
  bool all() const noexcept;
  size_t count() const noexcept;

  // --- modifiers ---
  CompactBitSet& set(size_t pos, bool value = true);
  CompactBitSet& set() noexcept;
  CompactBitSet& reset(size_t pos);
  CompactBitSet& reset() noexcept;

  CompactBitSet operator~() const noexcept;

  // --- bitwise in-place ---
  CompactBitSet& operator&=(const CompactBitSet& o) noexcept;
  CompactBitSet& operator|=(const CompactBitSet& o) noexcept;
  CompactBitSet& operator^=(const CompactBitSet& o) noexcept;

  bool operator==(const CompactBitSet&) const noexcept = default;
  bool operator!=(const CompactBitSet&) const noexcept = default;
  CompactBitSet operator&(const CompactBitSet&) const noexcept;
  CompactBitSet operator|(const CompactBitSet&) const noexcept;
  CompactBitSet operator^(const CompactBitSet&) const noexcept;

  // Custom methods, not in std::bitset

  // Returns an object that you can do a range-based for-loop over to iterate over the indices of
  // the set bits
  auto on_indices() const;

  // Returns an object that you can do a range-based for-loop over to iterate over the indices of
  // the unset bits
  auto off_indices() const;

  // Let A be a sorted array of integers k such that (*this)[k] is true.
  //
  // Returns A[n].
  //
  // Undefined behavior if this array access would be out of bounds.
  int get_nth_on_index(size_t n) const;

  // Returns the number of nonnegative integers k<n such that (*this)[k] is true
  int count_on_indices_before(size_t n) const;

  // Let A be the set of integers k such that (*this)[k] is true.
  //
  // Selects an integer uniformly at random from A.
  //
  // Undefined behavior is A is empty.
  int choose_random_on_index() const;

  // Let A be the set of integers k such that (*this)[k] is true.
  //
  // Selects a random subset S of size n from A.
  //
  // Sets (*this)[k] to false for each k in S.
  //
  // Undefined behavior is A has size < n.
  void randomly_zero_out(int n);

  // Returns a string of 0 and 1 chars, with (*this)[k] as the k'th char of the string.
  //
  // Note that this is the opposite order from std::bitset<N>::to_string()
  std::string to_string_natural() const;
};

}  // namespace util

#include "inline/util/CompactBitSet.inl"
