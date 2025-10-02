#include "util/CompactBitSet.hpp"

#include "util/Random.hpp"

#include <bit>
#include <limits>
#include <stdexcept>
#include <vector>

namespace util {

namespace detail {

// Here we have some helper classes used to support on_indices() and off_indices().

// Generic template (two specializations below)
template <bool WantOn, size_t M, size_t B>
struct BitIndexRange;

// ===================== One-word specialization (M == 1) =====================
template <bool WantOn, size_t B>
struct BitIndexRange<WantOn, 1, B> {
  using int_t = CompactBitSet<B>::int_t;

  struct It {
    int_t rem;      // remaining target bits as 1s (already tail-masked)
    uint8_t cur;    // current index, or nbits for end
    uint8_t nbits;  // total size

    It(int_t rem_, uint8_t nbits_, bool is_end)
        : rem(is_end ? int_t(0) : rem_), cur(is_end ? nbits_ : 0), nbits(nbits_) {
      if (!is_end) {
        if (rem) {
          cur = (int_t)std::countr_zero(rem);
          rem = int_t(rem & (rem - 1));
        } else {
          cur = nbits;
        }
      }
    }
    size_t operator*() const { return cur; }
    It& operator++() {
      if (!rem) {
        cur = nbits;
        return *this;
      }
      cur = std::countr_zero(rem);
      rem = int_t(rem & (rem - 1));
      return *this;
    }
    bool operator==(const It& o) const { return cur == o.cur && nbits == o.nbits && rem == o.rem; }
    bool operator!=(const It& o) const { return !(*this == o); }
  };

  int_t rem0;  // initial target-bit mask (set bits for ON, unset bits for OFF)
  uint8_t nbits;

  It begin() const { return It(rem0, nbits, /*is_end=*/false); }
  It end() const { return It(int_t(0), nbits, /*is_end=*/true); }
};

// ===================== Multi-word specialization (M > 1) =====================
template <bool WantOn, size_t M, size_t B>
struct BitIndexRange {
  static_assert(M > 1, "Use BitIndexRange<*,1,*> for the one-word case");
  static_assert(B == 64, "Multi-word CompactBitSet uses 64-bit chunks");

  struct It {
    const uint64_t* words;  // pointer to words[0]
    size_t w;               // next word to load
    size_t cw;              // word index for current 'rem'
    uint64_t rem;           // remaining target bits in current word
    size_t cur;             // current global bit index or nbits (end)
    size_t nbits;           // total size
    uint64_t last_mask;     // mask for tail bits in last word

    It(const uint64_t* p, size_t n, uint64_t tail, bool is_end)
        : words(p),
          w(is_end ? M : 0),
          cw(0),
          rem(0),
          cur(is_end ? n : 0),
          nbits(n),
          last_mask(tail) {
      if (!is_end) advance_to_next();
    }

    void advance_to_next() {
      while (true) {
        if (rem) {
          unsigned tz = std::countr_zero(rem);
          cur = cw * B + tz;  // B == 64 here
          rem &= (rem - 1);
          return;
        }
        if (w >= M) {
          cur = nbits;
          return;
        }

        uint64_t v = (WantOn ? words[w] : ~words[w]);
        if constexpr (!WantOn) {
          if (w + 1 == M) v &= last_mask;
        }

        cw = w;
        ++w;
        rem = v;  // if zero, loop pulls next word
      }
    }

    size_t operator*() const { return cur; }
    It& operator++() {
      advance_to_next();
      return *this;
    }
    bool operator==(const It& o) const { return cur == o.cur && nbits == o.nbits; }
    bool operator!=(const It& o) const { return !(*this == o); }
  };

  const uint64_t* words;
  size_t nbits;
  uint64_t last_mask;

  It begin() const { return It(words, nbits, last_mask, /*is_end=*/false); }
  It end() const { return It(words, nbits, last_mask, /*is_end=*/true); }
};

}  // namespace detail

template <size_t N>
constexpr CompactBitSet<N>::int_t CompactBitSet<N>::all_ones() noexcept {
  return std::numeric_limits<int_t>::max();
}

template <size_t N>
constexpr CompactBitSet<N>::int_t CompactBitSet<N>::tail_mask() noexcept {
  // Mask for the last word to zero out bits above N.
  if constexpr (M == 1) {
    constexpr size_t used = N % B;
    if (used == 0) return all_ones();
    return (int_t(1) << used) - int_t(1);
  } else {
    constexpr size_t used = N % 64;
    if (used == 0) return all_ones();
    // int_t is uint64_t here
    return (int_t(1) << used) - int_t(1);
  }
}

template <size_t N>
constexpr void CompactBitSet<N>::bounds_check(size_t pos) {
#ifdef NDEBUG
  return;
#endif
  if (pos >= N) throw std::out_of_range("CompactBitSet index");
}

template <size_t N>
constexpr void CompactBitSet<N>::mask_tail() noexcept {
  storage_.back() &= tail_mask();
}

template <size_t N>
constexpr size_t CompactBitSet<N>::size() noexcept {
  return N;
}

template <size_t N>
bool CompactBitSet<N>::operator[](size_t pos) const {
  return test(pos);
}

template <size_t N>
bool CompactBitSet<N>::test(size_t pos) const {
  bounds_check(pos);
  if constexpr (M == 1) {
    return (storage_[0] >> pos) & int_t(1);
  } else {
    const size_t w = pos / B;
    const size_t b = pos % B;
    return (storage_[w] >> b) & 1ull;
  }
}

template <size_t N>
bool CompactBitSet<N>::any() const noexcept {
  for (auto w : storage_)
    if (w) return true;
  return false;
}

template <size_t N>
bool CompactBitSet<N>::none() const noexcept {
  return !any();
}

template <size_t N>
bool CompactBitSet<N>::all() const noexcept {
  if constexpr (M == 1) {
    return storage_[0] == tail_mask();
  } else {
    // all full words except last must be all ones; last must match tail_mask
    for (size_t i = 0; i + 1 < M; ++i)
      if (storage_[i] != all_ones()) return false;
    return storage_.back() == tail_mask();
  }
}

template <size_t N>
size_t CompactBitSet<N>::count() const noexcept {
  if constexpr (M == 1) {
    return std::popcount(storage_[0]);
  } else {
    // uint64_t chunks
    size_t c = 0;
    for (size_t i = 0; i < M; ++i) {
      c += std::popcount(storage_[i]);
    }
    return c;
  }
}

// --- modifiers ---
template <size_t N>
CompactBitSet<N>& CompactBitSet<N>::set(size_t pos, bool value) {
  bounds_check(pos);
  if constexpr (M == 1) {
    const int_t bit = int_t(1) << pos;
    if (value) {
      storage_[0] |= bit;
    } else {
      storage_[0] &= ~bit;
    }
  } else {
    const size_t w = pos / 64;
    const size_t b = pos % 64;
    const uint64_t bit = 1ull << b;
    if (value)
      storage_[w] |= bit;
    else
      storage_[w] &= ~bit;
  }
  return *this;
}

template <size_t N>
CompactBitSet<N>& CompactBitSet<N>::set() noexcept {
  for (auto& w : storage_) w = all_ones();
  mask_tail();
  return *this;
}

template <size_t N>
CompactBitSet<N>& CompactBitSet<N>::reset(size_t pos) {
  return set(pos, false);
}

template <size_t N>
CompactBitSet<N>& CompactBitSet<N>::reset() noexcept {
  for (auto& w : storage_) w = 0;
  return *this;
}

template <size_t N>
CompactBitSet<N> CompactBitSet<N>::operator~() const noexcept {
  CompactBitSet r;
  for (size_t i = 0; i < M; ++i) r.storage_[i] = ~storage_[i];
  r.mask_tail();
  return r;
}

// --- bitwise in-place ---
template <size_t N>
CompactBitSet<N>& CompactBitSet<N>::operator&=(const CompactBitSet& o) noexcept {
  for (size_t i = 0; i < M; ++i) storage_[i] &= o.storage_[i];
  return *this;
}
template <size_t N>
CompactBitSet<N>& CompactBitSet<N>::operator|=(const CompactBitSet& o) noexcept {
  for (size_t i = 0; i < M; ++i) storage_[i] |= o.storage_[i];
  return *this;
}
template <size_t N>
CompactBitSet<N>& CompactBitSet<N>::operator^=(const CompactBitSet& o) noexcept {
  for (size_t i = 0; i < M; ++i) storage_[i] ^= o.storage_[i];
  return *this;
}

// --- bitwise const ---

template <size_t N>
CompactBitSet<N> CompactBitSet<N>::operator&(const CompactBitSet& o) const noexcept {
  CompactBitSet r;
  for (std::size_t i = 0; i < M; ++i) r.storage_[i] = storage_[i] & o.storage_[i];
  return r;
}

template <size_t N>
CompactBitSet<N> CompactBitSet<N>::operator|(const CompactBitSet& o) const noexcept {
  CompactBitSet r;
  for (std::size_t i = 0; i < M; ++i) r.storage_[i] = storage_[i] | o.storage_[i];
  return r;
}

template <size_t N>
CompactBitSet<N> CompactBitSet<N>::operator^(const CompactBitSet& o) const noexcept {
  CompactBitSet r;
  for (std::size_t i = 0; i < M; ++i) r.storage_[i] = storage_[i] ^ o.storage_[i];
  return r;
}

// Custom methods, not in std::bitset

template <size_t N>
auto CompactBitSet<N>::on_indices() const {
  if constexpr (M == 1) {
    return detail::BitIndexRange<true, 1, B>{storage_[0], size()};
  } else {
    return detail::BitIndexRange<true, M, B>{&storage_[0], size(), tail_mask()};
  }
}

template <size_t N>
auto CompactBitSet<N>::off_indices() const {
  if constexpr (M == 1) {
    int_t off = int_t(~storage_[0]) & tail_mask();
    return detail::BitIndexRange<false, 1, B>{off, size()};
  } else {
    return detail::BitIndexRange<false, M, B>{&storage_[0], size(), tail_mask()};
  }
}

// ----------------------------------------------
// get_nth_on_index: k-th set bit (0-based)
// ----------------------------------------------
template <size_t N>
int CompactBitSet<N>::get_nth_on_index(size_t n) const {
  size_t want = n;
  if constexpr (M == 1) {
    int_t v = storage_[0];
#ifndef NDEBUG
    int pc = std::popcount(v);
    if (n >= pc) throw std::out_of_range("get_nth_on_index");
#endif
    while (want--) v &= (v - 1);  // clear 'want' lowest set bits
    return std::countr_zero(v);   // index in [0, B)
  } else {
    int base = 0;
    for (size_t w = 0; w < M; ++w, base += 64) {
      auto v = storage_[w];
      const unsigned pc = std::popcount(v);
      if (want >= pc) {
        want -= pc;
        continue;
      }
      // 'want' is inside this word: select the want-th 1-bit
      while (want--) v &= (v - 1);
      return base + std::countr_zero(v);
    }
    // UB by spec if out of bounds_check:
#if defined(NDEBUG)
    __builtin_unreachable();
#else
    throw std::out_of_range("get_nth_on_index: n out of range");
#endif
  }
}

// --------------------------------------------------
// count_on_indices_before(n)
// --------------------------------------------------
template <size_t N>
int CompactBitSet<N>::count_on_indices_before(size_t n) const {
  if (n == 0) return 0;

  if constexpr (M == 1) {
    int_t v = storage_[0];
    if (n < B) {
      v &= (int_t(1) << n) - 1;
    }
    return std::popcount(v);
  } else {
    if (n > N) n = N;
    size_t full_words = n / 64;
    size_t rem_bits = n % 64;
    int total = 0;
    for (size_t w = 0; w < full_words; ++w) {
      total += std::popcount(storage_[w]);
    }
    if (rem_bits) {
      uint64_t v = storage_[full_words];
      v &= (1ull << rem_bits) - 1;
      total += std::popcount(v);
    }
    return total;
  }
}

// ----------------------------------------------
// Random selection: ON index (uniform)
// ----------------------------------------------
template <size_t N>
int CompactBitSet<N>::choose_random_on_index() const {
  size_t n = util::Random::uniform_sample(size_t(0), count());
  return get_nth_on_index(n);
}

// ------------------------------------------------------
// randomly_zero_out(n): choose n distinct ON bits, clear
// ------------------------------------------------------
template <size_t N>
void CompactBitSet<N>::randomly_zero_out(int n) {
  // reservoir sampling
  std::vector<int> reservoir;
  reservoir.reserve(n);

  int k = 0;
  for (int i : on_indices()) {
    ++k;
    if ((int)reservoir.size() < n) {
      reservoir.push_back(i);
    } else {
      int j = util::Random::uniform_sample(int(0), k);
      if (j < n) {
        reservoir[j] = i;
      }
    }
  }

  for (int i : reservoir) {
    reset(i);
  }
}

// ------------------------------------------------------
// to_string_natural(): s[k] is bit k ('0'/'1')
// ------------------------------------------------------
template <size_t N>
std::string CompactBitSet<N>::to_string_natural() const {
  std::string s(size(), '0');
  if constexpr (M == 1) {
    auto v = storage_[0];
    while (v) {
      unsigned tz = std::countr_zero(v);
      s[tz] = '1';
      v &= (v - 1);
    }
  } else {
    size_t base = 0;
    for (size_t w = 0; w < M; ++w, base += 64) {
      auto v = storage_[w];
      while (v) {
        unsigned tz = std::countr_zero(v);
        s[base + tz] = '1';
        v &= (v - 1);
      }
    }
  }
  return s;
}

}  // namespace util
