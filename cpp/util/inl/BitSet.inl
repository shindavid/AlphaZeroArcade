#include <util/BitSet.hpp>

#include <util/Random.hpp>

#include <random>

namespace util {

/*
 * Adapted from: https://stackoverflow.com/a/37460774/543913
 *
 * TODO: optimize by using custom implementation powered by c++20's <bits> module.
 */
template<int N>
int BitSet<N>::choose_random_set_bit() const {
  // Adapted from: https://stackoverflow.com/a/37460774/543913
  int c = 1 + Random::uniform_draw(0, int(this->count()));
  int p = 0;
  for (; c; ++p) c -= (*this)[p];
  return p - 1;
}

template<int N>
eigen_util::fixed_tensor_t<float, Eigen::Sizes<N>> BitSet<N>::to_float_tensor() const {
  eigen_util::fixed_tensor_t<float, Eigen::Sizes<N>> tensor;
  for (int i = 0; i < N; ++i) {
    tensor(i) = (*this)[i];
  }
  return tensor;
}

}  // namespace util
