#include <util/Random.hpp>

#include <ctime>

namespace util {


template<typename T>
inline T Random::uniform_draw(T lower, T upper) {
  Random* random = instance();
  std::uniform_int_distribution<T> dist{lower, upper};
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
