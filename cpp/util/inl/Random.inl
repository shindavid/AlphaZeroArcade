#include <util/Random.hpp>

#include <ctime>

namespace util {

template<typename T, typename U>
inline auto Random::uniform_sample(T lower, U upper) {
  using V = decltype(std::declval<T>() + std::declval<U>());
  std::uniform_int_distribution<V> dist{(V)lower, (V)(upper - 1)};
  return dist(instance()->prng_);
}

template<typename FloatType>
FloatType Random::uniform_real(FloatType left, FloatType right) {
  std::uniform_real_distribution<FloatType> dist(left, right);
  return dist(instance()->prng_);
}


template<std::random_access_iterator T>
void Random::shuffle(T begin, T end) {
  return std::shuffle(begin, end, instance()->prng_);
}

template<typename IntType, typename InputIt>
inline IntType Random::weighted_sample(InputIt begin, InputIt end) {
  std::discrete_distribution<IntType> dist(begin, end);
  return dist(instance()->prng_);
}

template<typename InputIt>
inline int Random::weighted_sample(InputIt begin, InputIt end) {
  return weighted_sample<int>(begin, end);
}

inline Random* Random::instance() {
  if (!instance_) {
    instance_ = new Random();
  }
  return instance_;
}

inline Random::Random() : prng_(std::time(nullptr)) {}

}  // namespace util
