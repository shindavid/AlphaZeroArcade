#pragma once
/*
 * util::BitSet<N> is like std::bitset<N>. It extends the interface by providing a way to iterate over the set bits,
 * along with various other auxiliary helpers.
 *
 * For now, the implementation just wraps std::bitset<N>, and the set-bits-iterator is no more efficient than
 * stepping through all N bits and branching. Later, we can optimize the implementation.
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
public:
  struct SetBitsIterator {
  public:
    SetBitsIterator(const BitSet* bitset, int index) : bitset_(bitset), index_(index) { skip_to_next(); }
    bool operator==(SetBitsIterator other) const { return index_ == other.index_; }
    bool operator!=(SetBitsIterator other) const { return index_ != other.index_; }
    int operator*() const { return index_;}
    SetBitsIterator& operator++() { index_++; skip_to_next(); return *this; }
    SetBitsIterator operator++(int) { SetBitsIterator tmp = *this; ++(*this); return tmp; }
  private:
    void skip_to_next() { while (index_ < N && !(*bitset_)[index_]) index_++; }
    const BitSet* bitset_;
    int index_;
  };

  SetBitsIterator begin() const { return SetBitsIterator(this, 0); }
  SetBitsIterator end() const { return SetBitsIterator(this, N); }

  int choose_random_set_bit() const;
  void to_float_vector(Eigen::Vector<float, N>&) const;
};

/*
namespace detail {

template<size_t max_value>
struct uint_type_selector {
    using type = std::conditional_t<
            (max_value<=8),
            uint8_t,
            std::conditional_t<
                    (max_value<=16),
                    uint16_t,
                    std::conditional_t<
                            (max_value<=32),
                            uint32_t,
                            uint64_t>>>;
};
template<size_t max_value> using uint_type_selector_t = typename uint_type_selector<max_value>::type;

}  // namespace detail

template <int N>
class __attribute__((__packed__)) BitSet {
public:
    using page_t = detail::uint_type_selector_t<N>;
    static const int kBitsPerPage = sizeof(page_t) * 8;
    static const int kNumPages = 1 + (N-1) / 64;

    struct bit_ref_t {
        bit_ref_t(page_t& page, uint8_t index) : page_(page), index_(index) {};
        operator bool() const { return page_ & (1UL << index_); }
        bool operator=(bool b) {
            // https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
            page_ ^= (-b ^ page_) & (1UL << index_);
            return b;
        }
    private:
        page_t& page_;
        uint8_t index_;
    };

    struct const_bit_ref_t {
        const_bit_ref_t(const page_t& page, uint8_t index) : page_(page), index_(index) {};
        operator bool() const { return page_ & (1UL << index_); }
    private:
        const page_t& page_;
        uint8_t index_;
    };

    bit_ref_t operator[](int k) { return bit_ref_t(pages_[k / kBitsPerPage], k % kBitsPerPage); }
    const_bit_ref_t operator[](int k) const { return const_bit_ref_t(pages_[k / kBitsPerPage], k % kBitsPerPage); }

private:
    page_t pages_[kNumPages] = {};
};
*/

}  // namespace util

template<typename T> struct is_bit_set { static const bool value = false; };
template<int N> struct is_bit_set<util::BitSet<N>> { static const bool value = true; };
template<typename T> inline constexpr bool is_bit_set_v = is_bit_set<T>::value;
template <typename T> concept is_bit_set_c = is_bit_set_v<T>;

#include <util/inl/BitSet.inl>
