#pragma once

#include <random>

/*
 * A wrapper around STL's random machinery.
 */

namespace util {

class Random {
public:
  /*
   * Uniformly randomly picks a value in the half-open range [lower, upper).
   */
  template<typename T, typename U> static auto uniform_draw(T lower, U upper);

private:
  static Random* instance();
  Random();

  static Random* instance_;
  std::mt19937 prng_;
};

}  // namespace util

#include <util/inl/Random.inl>
