#include "util/BoostUtil.hpp"
#include "util/CppUtil.hpp"
#include "util/Random.hpp"

#include <algorithm>
#include <ctime>

namespace util {

inline auto Random::Params::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Random options");

  return desc.template add_option<"seed">(po::value<int>(&seed)->default_value(seed),
                                          "random seed (default: 0 means seed with current time)");
}

inline void Random::init(const Params& params) {
  if (params.seed) {
    set_seed(params.seed);
  }
}

inline void Random::set_seed(int seed) { default_prng().seed(seed); }

template <std::integral T, std::integral U>
inline auto Random::uniform_sample(std::mt19937& prng, T lower, U upper) {
  if (lower >= upper) {
    throw std::runtime_error("Random::uniform_sample() - invalid range");
  }
  using V = std::common_type_t<T, U>;
  std::uniform_int_distribution<V> dist{(V)lower, (V)(upper - 1)};
  return dist(prng);
}

template <std::integral T, std::integral U>
inline auto Random::uniform_sample(T lower, U upper) {
  return uniform_sample(default_prng(), lower, upper);
}

template <typename FloatType>
FloatType Random::uniform_real(std::mt19937& prng, FloatType left, FloatType right) {
  if (left >= right) {
    throw std::runtime_error("Random::uniform_real() - invalid range");
  }
  std::uniform_real_distribution<FloatType> dist(left, right);
  return dist(prng);
}

template <typename FloatType>
FloatType Random::uniform_real(FloatType left, FloatType right) {
  return uniform_real(default_prng(), left, right);
}

template <typename RealType>
RealType Random::exponential(std::mt19937& prng, RealType lambda) {
  std::exponential_distribution<RealType> dist(lambda);
  return dist(prng);
}

template <typename RealType>
RealType Random::exponential(RealType lambda) {
  return exponential(default_prng(), lambda);
}

template <std::random_access_iterator T>
void Random::shuffle(std::mt19937& prng, T begin, T end) {
  std::shuffle(begin, end, prng);
}

template <std::random_access_iterator T>
void Random::shuffle(T begin, T end) {
  shuffle(default_prng(), begin, end);
}

template <std::random_access_iterator T>
void Random::chunked_shuffle(std::mt19937& prng, T begin, T end, int chunk_size) {
  int c = chunk_size;
  int n = (end - begin) / c;

  // Fisher–Yates shuffle on groups.
  for (int i = n - 1; i > 0; i--) {
    std::uniform_int_distribution<int> dist(0, i);
    int j = dist(prng);
    if (i != j) {
      T a = begin + i * c;
      T b = begin + j * c;
      std::swap_ranges(a, a + c, b);
    }
  }
}

template <std::random_access_iterator T>
void Random::chunked_shuffle(T begin, T end, int chunk_size) {
  chunked_shuffle(default_prng(), begin, end, chunk_size);
}

template <typename InputIt>
inline int Random::weighted_sample(std::mt19937& prng, InputIt begin, InputIt end) {
  std::discrete_distribution<int> dist(begin, end);
  return dist(prng);
}

template <typename InputIt>
inline int Random::weighted_sample(InputIt begin, InputIt end) {
  return weighted_sample(default_prng(), begin, end);
}

template <typename InputIt>
void Random::zero_out(std::mt19937& prng, InputIt begin, InputIt end, size_t n) {
  // reservoir sampling
  std::vector<InputIt> reservoir;
  reservoir.reserve(n);

  size_t num_non_zero_entries_found = 0;
  for (InputIt it = begin; it != end; ++it) {
    if (!(*it)) continue;  // skip zero elements

    ++num_non_zero_entries_found;
    if (reservoir.size() < n) {
      reservoir.push_back(it);
    } else {
      size_t j = uniform_sample(prng, size_t(0), num_non_zero_entries_found);
      if (j < n) {
        reservoir[j] = it;
      }
    }
  }

  for (InputIt it : reservoir) {
    *it = 0;
  }
}

template <typename InputIt>
void Random::zero_out(InputIt begin, InputIt end, size_t n) {
  zero_out(default_prng(), begin, end, n);
}

inline std::mt19937& Random::default_prng() {
  static std::mt19937 prng(std::time(nullptr));
  return prng;
}

}  // namespace util
