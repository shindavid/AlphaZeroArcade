#include <util/BitSet.hpp>

#include <algorithm>
#include <cassert>

#include <util/Asserts.hpp>
#include <util/Exception.hpp>
#include <util/Random.hpp>

namespace bitset_util {

namespace detail {

enum IterType { kSet, kUnset };

template <size_t N, IterType type>
struct Wrapper {
  using bitset_t = std::bitset<N>;

  struct Iterator {
   public:
    Iterator(const bitset_t* bitset, int index) : bitset_(bitset), index_(index) { skip_to_next(); }
    bool operator==(Iterator other) const { return index_ == other.index_; }
    bool operator!=(Iterator other) const { return index_ != other.index_; }
    int operator*() const { return index_; }
    Iterator& operator++() {
      index_++;
      skip_to_next();
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

   private:
    static constexpr bool use(bool b) { return (type == kUnset) ^ b; }
    void skip_to_next() {
      while (index_ < N && !use((*bitset_)[index_])) index_++;
    }
    const bitset_t* bitset_;
    size_t index_;
  };

  Wrapper(const bitset_t* bitset) : bitset_(bitset) {}

  Iterator begin() const { return Iterator(bitset_, 0); }
  Iterator end() const { return Iterator(bitset_, N); }

  const bitset_t* bitset_;
};

}  // namespace detail

template <size_t N>
auto on_indices(const std::bitset<N>& bitset) {
  return detail::Wrapper<N, detail::kSet>(&bitset);
}

template <size_t N>
auto off_indices(const std::bitset<N>& bitset) {
  return detail::Wrapper<N, detail::kUnset>(&bitset);
}

template <size_t N>
int get_nth_on_index(const std::bitset<N>& bitset, int n) {
  for (int k : on_indices(bitset)) {
    if (n == 0) return k;
    n--;
  }
  throw util::Exception("bitset_util::get_nth_on_index: n is out of bounds [%s] [%d]",
                        bitset.to_string().c_str(), n);
}

template <size_t N>
int count_on_indices_before(const std::bitset<N>& bitset, int i) {
  int count = 0;
  for (int k : on_indices(bitset)) {
    if (k >= i) break;
    count++;
  }
  return count;
}

/*
 * Adapted from: https://stackoverflow.com/a/37460774/543913
 *
 * TODO: optimize by using custom implementation powered by c++20's <bits> module.
 */
template <size_t N>
int choose_random_on_index(const std::bitset<N>& bitset) {
  int upper = bitset.count();
  util::release_assert(upper > 0);
  int c = 1 + util::Random::uniform_sample(0, upper);
  int p = 0;
  for (; c; ++p) c -= bitset[p];
  return p - 1;
}

/*
 * Adapted from: https://stackoverflow.com/a/37460774/543913
 *
 * TODO: optimize by using custom implementation powered by c++20's <bits> module.
 */
template <size_t N>
int choose_random_off_index(const std::bitset<N>& bitset) {
  int upper = N - bitset.count();
  util::release_assert(upper > 0);
  int c = 1 + util::Random::uniform_sample(0, upper);
  int p = 0;
  for (; c; ++p) c -= not bitset[p];
  return p - 1;
}

template <size_t N>
void randomly_zero_out(std::bitset<N>& bitset, int n) {
  // reservoir sampling
  std::vector<int> reservoir;
  reservoir.reserve(n);

  int k = 0;
  for (int i : on_indices(bitset)) {
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
    bitset[i] = 0;
  }
}

template <size_t N>
std::string to_string(const std::bitset<N>& bitset) {
  std::string s = bitset.to_string();
  std::reverse(s.begin(), s.end());
  return s;
}

}  // namespace bitset_util
