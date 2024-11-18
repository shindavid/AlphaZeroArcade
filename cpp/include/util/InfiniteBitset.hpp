#pragma once

#include <boost/dynamic_bitset.hpp>

#include <cstdint>

namespace util {

/*
 * InfiniteBitset is like boost::dynamic_bitset<>, with a key difference: if we attempt to access an
 * out-of-bounds index, it will automatically resize to accommodate the new index.
 */
class InfiniteBitset {
 public:
  InfiniteBitset(int initial_size = 64) { bits_.resize(initial_size); }

  using bitset_t = boost::dynamic_bitset<>;
  using reference = bitset_t::reference;

  reference operator[](size_t pos);
  void set(size_t n);
  void reset();
  size_t count() const;

 protected:
  void auto_resize(size_t pos);

  boost::dynamic_bitset<> bits_;
};

}  // namespace util

#include <inline/util/InfiniteBitset.inl>
