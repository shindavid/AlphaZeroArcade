#include <util/Random.hpp>

#include <ctime>

namespace util {

template<typename T, typename U>
inline auto Random::uniform_draw(T lower, U upper) {
  Random* random = instance();
  using V = decltype(std::declval<T>() + std::declval<U>());
  std::uniform_int_distribution<V> dist{(V)lower, (V)upper};
  return dist(random->prng_);
}

inline Random* Random::instance() {
  if (!instance_) {
    instance_ = new Random();
  }
  return instance_;
}

inline Random::Random() : prng_(std::time(nullptr)) {}

}  // namespace util
