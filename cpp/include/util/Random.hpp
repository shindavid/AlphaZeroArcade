#pragma once

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
   */
  template <typename T, typename U>
  static auto uniform_sample(T lower, U upper);

  /*
   * Produces a random real value in the range [left, right)
   */
  template <typename FloatType>
  static FloatType uniform_real(FloatType left, FloatType right);

  /*
   * Produces a random real value using the exponential distribution with mean 1/lambda
   */
  template <typename RealType>
  static RealType exponential(RealType lambda);

  template <std::random_access_iterator T>
  static void shuffle(T begin, T end);

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
   */
  template <typename IntType, typename InputIt>
  static IntType weighted_sample(InputIt begin, InputIt end);
  template <typename InputIt>
  static int weighted_sample(InputIt begin, InputIt end);  // default IntType=int

  /*
   * Randomly zeroes out n nonzero elements in the range [begin, end).
   *
   * Requires that [begin, end) contains at least n nonzero elements.
   */
  template <typename InputIt>
  static void zero_out(InputIt begin, InputIt end, size_t n);

 private:
  static Random* instance();
  Random();

  static Random* instance_;
  std::mt19937 prng_;
};

}  // namespace util

#include <inline/util/Random.inl>
