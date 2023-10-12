#pragma once

#include <iterator>
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
  template<typename T, typename U> static auto uniform_sample(T lower, U upper);

  /*
   * Produces a random real value in the range [left, right)
   */
  template<typename FloatType> static FloatType uniform_real(FloatType left, FloatType right);

  /*
   * Produces a random real value using the exponential distribution with mean 1/lambda
   */
  template<typename RealType> static RealType exponential(RealType lambda);

  template<std::random_access_iterator T> static void shuffle(T begin, T end);

  /*
   * Given an array A of n values, produces a random integer on the interval [0, n), where integer i is chosen
   * with probability proportional to A[i].
   *
   * The begin/end of the array is passed in as the two arguments.
   *
   * Example:
   *
   * std::array<float, 3> arr = {1, 2, 3};
   * int k = util::Random::weighted_sample(arr.begin(), arr.end());
   */
  template<typename IntType, typename InputIt> static IntType weighted_sample(InputIt begin, InputIt end);
  template<typename InputIt> static int weighted_sample(InputIt begin, InputIt end);  // default IntType=int

  /*
   * Randomly zeroes out n nonzero elements in the range [begin, end).
   *
   * Requires that [begin, end) contains at least n nonzero elements.
   */
  template<typename InputIt> static void zero_out(InputIt begin, InputIt end, size_t n);

  static void set_seed(int seed);

 private:
  static Random* instance();
  Random();

  static Random* instance_;
  std::mt19937 prng_;
};

}  // namespace util

#include <util/inl/Random.inl>
