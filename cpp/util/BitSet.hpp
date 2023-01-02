#pragma once
/*
 * util::BitSet<N> is like std::bitset<N>. It extends the interface by providing a way to iterate over the set bits,
 * along with various other auxiliary helpers.
 *
 * TODO: util::BitSet doesn't really deserve to be a class in itself. We can work directly with std::bitset instead,
 * and just provide a function that accepts std::bitset and returns something that can be iterated over.
 *
 * Usage:
 *
 * using BitSet = util::BitSet<4>;
 *
 * BitSet b;
 * b[1] = true;
 *
 * for (int k : b.set_bits()) {
 *   printf("set %d\n", k);
 * }
 *
 * for (int k : b.unset_bits()) {
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
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <util/EigenUtil.hpp>

namespace util {

/*
 * TODO: optimize by using custom implementation powered by c++20's <bits> module. As-is, finding all the set-bits of
 * an empty BitSet<256> entails checking 256 bits, when in fact only 4 uint64_t's should need to be checked, so this
 * implementation is ~64x inefficient.
 *
 * Default constructor sets all bits to zero.
 */
template<int N>
class BitSet : public std::bitset<N> {
protected:
  enum IterType {
    kSet,
    kUnset
  };

  template <IterType type>
  struct Wrapper {
    struct Iterator {
    public:
      Iterator(const BitSet* bitset, int index) : bitset_(bitset), index_(index) { skip_to_next(); }
      bool operator==(Iterator other) const { return index_ == other.index_; }
      bool operator!=(Iterator other) const { return index_ != other.index_; }
      int operator*() const { return index_;}
      Iterator& operator++() { index_++; skip_to_next(); return *this; }
      Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
    private:
      static constexpr bool use(bool b) { return (type==kUnset) ^ b; }
      void skip_to_next() { while (index_ < N && !use((*bitset_)[index_])) index_++; }
      const BitSet* bitset_;
      int index_;
    };

    Wrapper(const BitSet* bitset) : bitset_(bitset) {}

    Iterator begin() const { return Iterator(bitset_, 0); }
    Iterator end() const { return Iterator(bitset_, N); }

    const BitSet* bitset_;
  };

public:
  auto set_bits() const { return Wrapper<kSet>(this); }
  auto unset_bits() const { return Wrapper<kUnset>(this); }

  int choose_random_set_bit() const;
  template<typename T> void to_array(T* arr) const;
};

}  // namespace util

template<typename T> struct is_bit_set { static const bool value = false; };
template<int N> struct is_bit_set<util::BitSet<N>> { static const bool value = true; };
template<typename T> inline constexpr bool is_bit_set_v = is_bit_set<T>::value;
template <typename T> concept is_bit_set_c = is_bit_set_v<T>;

#include <util/inl/BitSet.inl>
