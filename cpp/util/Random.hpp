#pragma once

#include <random>

/*
 * A wrapper around STL's random machinery.
 */

namespace util {

class Random {
public:
  /*
   * Uniformly randomly picks an int in the closed range [lower, upper].
   */
  template<typename T>
  static T uniform_draw(T lower, T upper);

private:
  static Random* instance();
  Random();

  static Random* instance_;
  std::mt19937 prng_;
};

}  // namespace util

#include <util/RandomINLINES.cpp>
