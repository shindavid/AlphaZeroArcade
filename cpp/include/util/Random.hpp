#pragma once

#include <concepts>
#include <iterator>
#include <random>

/*
 * A wrapper around STL's random machinery.
 *
 * To use with a timer-based seed, just directly use any of the functions, such as
 * util::Random::uniform_sample().
 *
 * To initialize with a specific seed, first call:
 *
 * util::Random::set_seed(seed);
 *
 * To add a cmdline option to set the seed, do:
 *
 * util::Random::Params random_params;
 *
 * namespace po2 = boost_util::program_options;
 * po2::options_description raw_desc("General options");
 * auto desc = raw_desc.add(random_params.make_options_description());
 * po2::parse_args(desc, ac, av);
 *
 * util::Random::init(random_params);
 *
 * Each of the random functions in this class has 2 variants: one that accepts a std::mt19937
 * reference as the first argument, and one that doesn't. The latter uses the default prng
 * (controlled by the seed set above). The former allows you to maintain multiple independent
 * prngs, which can be useful in some cases.
 */
namespace util {

class Random {
 public:
  struct Params {
    auto make_options_description();

    int seed = 0;
  };

  static void init(const Params&);

  static void set_seed(int seed);

  /*
   * Uniformly randomly picks a value in the half-open range [lower, upper).
   *
   * T and U should be integral types, and lower must be less than upper.
   *
   * Uses prng as the random number generator.
   */
  template <std::integral T, std::integral U>
  static auto uniform_sample(std::mt19937& prng, T lower, U upper);

  /*
   * Uniformly randomly picks a value in the half-open range [lower, upper).
   *
   * T and U should be integral types, and lower must be less than upper.
   *
   * Uses the default prng.
   */
  template <std::integral T, std::integral U>
  static auto uniform_sample(T lower, U upper);

  /*
   * Produces a random real value in the range [left, right).
   *
   * Uses prng as the random number generator.
   */
  template <typename FloatType>
  static FloatType uniform_real(std::mt19937& prng, FloatType left, FloatType right);

  /*
   * Produces a random real value in the range [left, right).
   *
   * Uses the default prng.
   */
  template <typename FloatType>
  static FloatType uniform_real(FloatType left, FloatType right);

  /*
   * Produces a random real value using the exponential distribution with mean 1/lambda.
   *
   * Uses prng as the random number generator.
   */
  template <typename RealType>
  static RealType exponential(std::mt19937& prng, RealType lambda);

  /*
   * Produces a random real value using the exponential distribution with mean 1/lambda.
   *
   * Uses the default prng.
   */
  template <typename RealType>
  static RealType exponential(RealType lambda);

  template <std::random_access_iterator T>
  static void shuffle(std::mt19937& prng, T begin, T end);

  template <std::random_access_iterator T>
  static void shuffle(T begin, T end);

  template <std::random_access_iterator T>
  static void chunked_shuffle(std::mt19937& prng, T begin, T end, int chunk_size);

  template <std::random_access_iterator T>
  static void chunked_shuffle(T begin, T end, int chunk_size);

  /*
   * Given an array A of n values, produces a random integer on the interval [0, n), where integer i
   * is chosen with probability proportional to A[i].
   *
   * The begin/end of the array is passed in as the two arguments.
   *
   * Example:
   *
   * std::array<float, 3> arr = {1, 2, 3};
   * int k = util::Random::weighted_sample(arr.begin(), arr.end());
   *
   * Uses prng as the random number generator.
   */
  template <typename InputIt>
  static int weighted_sample(std::mt19937& prng, InputIt begin, InputIt end);

  /*
   * Given an array A of n values, produces a random integer on the interval [0, n), where integer i
   * is chosen with probability proportional to A[i].
   *
   * The begin/end of the array is passed in as the two arguments.
   *
   * Example:
   *
   * std::array<float, 3> arr = {1, 2, 3};
   * int k = util::Random::weighted_sample(arr.begin(), arr.end());
   *
   * Uses the default prng.
   */
  template <typename InputIt>
  static int weighted_sample(InputIt begin, InputIt end);

  /*
   * Randomly zeroes out n nonzero elements in the range [begin, end).
   *
   * Requires that [begin, end) contains at least n nonzero elements.
   *
   * Uses prng as the random number generator.
   */
  template <typename InputIt>
  static void zero_out(std::mt19937& prng, InputIt begin, InputIt end, size_t n);

  /*
   * Randomly zeroes out n nonzero elements in the range [begin, end).
   *
   * Requires that [begin, end) contains at least n nonzero elements.
   *
   * Uses the default prng.
   */
  template <typename InputIt>
  static void zero_out(InputIt begin, InputIt end, size_t n);

  static std::mt19937& default_prng();
};

}  // namespace util

#include "inline/util/Random.inl"
