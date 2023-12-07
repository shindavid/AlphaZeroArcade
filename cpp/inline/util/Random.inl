#include <util/Random.hpp>

#include <ctime>

#include <util/CppUtil.hpp>

namespace util {

template <typename T, typename U>
inline auto Random::uniform_sample(T lower, U upper) {
  if (lower >= upper) {
    throw std::runtime_error("Random::uniform_sample() - invalid range");
  }
  using V = decltype(std::declval<T>() + std::declval<U>());
  std::uniform_int_distribution<V> dist{(V)lower, (V)(upper - 1)};
  return dist(instance()->prng_);
}

template <typename FloatType>
FloatType Random::uniform_real(FloatType left, FloatType right) {
  if (left >= right) {
    throw std::runtime_error("Random::uniform_real() - invalid range");
  }
  std::uniform_real_distribution<FloatType> dist(left, right);
  return dist(instance()->prng_);
}

template <typename RealType>
RealType Random::exponential(RealType lambda) {
  std::exponential_distribution<RealType> dist(1.0 / lambda);
  return dist(instance()->prng_);
}

template <std::random_access_iterator T>
void Random::shuffle(T begin, T end) {
  return std::shuffle(begin, end, instance()->prng_);
}

template <typename IntType, typename InputIt>
inline IntType Random::weighted_sample(InputIt begin, InputIt end) {
  std::discrete_distribution<IntType> dist(begin, end);
  return dist(instance()->prng_);
}

template <typename InputIt>
inline int Random::weighted_sample(InputIt begin, InputIt end) {
  return weighted_sample<int>(begin, end);
}

template <typename InputIt>
void Random::zero_out(InputIt begin, InputIt end, size_t n) {
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
      size_t j = uniform_sample(size_t(0), num_non_zero_entries_found);
      if (j < n) {
        reservoir[j] = it;
      }
    }
  }

  for (InputIt it : reservoir) {
    *it = 0;
  }
}

inline void Random::set_seed(int seed) { instance()->prng_.seed(seed); }

inline Random* Random::instance() {
  if (!instance_) {
    instance_ = new Random();
  }
  return instance_;
}

inline Random::Random() : prng_(IS_MACRO_ENABLED(DETERMINISTIC_MODE) ? 1234 : std::time(nullptr)) {}

}  // namespace util
