#pragma once

#include <bit>
#include <cstdint>
#include <iterator>

namespace util {

// TinyBitSet is essentially std::bitset<64>, optimized for that fixed size.
//
// It includes a begin()/end() iterator interface, which allows for range-based for loops to
// iterate over the set bits.
class TinyBitSet {
 public:
  TinyBitSet() = default;
  explicit TinyBitSet(uint64_t bits) : bits_(bits) {}

  void set(size_t pos) { bits_ |= (uint64_t(1) << pos); }
  void reset(size_t pos) { bits_ &= ~(uint64_t(1) << pos); }
  bool test(size_t pos) const { return bits_ & (uint64_t(1) << pos); }
  void clear() { bits_ = 0; }
  bool empty() const { return bits_ == 0; }
  uint64_t raw() const { return bits_; }

  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = const size_t*;
    using reference = const size_t&;

    Iterator(uint64_t bits, size_t pos) : bits_(bits), pos_(pos) { advance_to_next(); }

    size_t operator*() const { return pos_; }

    Iterator& operator++() {
      bits_ &= ~(uint64_t(1) << pos_);  // clear current bit
      advance_to_next();
      return *this;
    }

    bool operator==(const Iterator& other) const {
      return bits_ == other.bits_ && pos_ == other.pos_;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }

   private:
    void advance_to_next() {
      if (bits_ == 0) {
        pos_ = 64;
        return;
      }
      pos_ = std::countr_zero(bits_);  // count trailing zeros
    }

    uint64_t bits_ = 0;
    size_t pos_ = 0;
  };

  Iterator begin() const { return Iterator(bits_, 0); }
  Iterator end() const { return Iterator(0, 64); }

 private:
  uint64_t bits_ = 0;
};

}  // namespace util
