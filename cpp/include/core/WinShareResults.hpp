#pragma once

#include <core/BasicTypes.hpp>
#include <util/EigenUtil.hpp>

#include <algorithm>

namespace core {

/*
 * WinShareResults can be used for p-player games with win/loss/draw outcomes with p>=2. It encodes
 * the results of a game as a length-p array of nonnegative floats summing to 1.
 *
 * If a player wins outright, their value is 1 and all other values are 0.
 *
 * If a player ties with k other players, each of those players has a value of 1/(k+1).
 *
 * For two-player games, it is generally better to use WinLossDrawResults, since the ability to
 * explicitly predict draws can be useful, and allows the value-loss to be driven down to 0.
 *
 * For games with more than two players, the analog of WinLossDrawResults would demand an output
 * of length 2^p - 1, which is not practical (particularly because many of those outcomes may be
 * sparsely represented in the training data). WinShareResults is a good compromise.
 */
template<int kNumPlayers>
struct WinShareResults {
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kNumPlayers>>;
  using ValueArray = eigen_util::FArray<kNumPlayers>;

  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor win(core::seat_index_t seat) {
    Tensor result;
    result.setZero();
    result(seat) = 1;
    return result;
  }

  static ValueArray to_value_array(const Tensor& t) {
    return eigen_util::reinterpret_as_array(t);
  }

  static void left_rotate(Tensor& t, core::seat_index_t s) {
    ValueArray& v = eigen_util::reinterpret_as_array(t);
    eigen_util::left_rotate(v, s);
  }

  static void right_rotate(Tensor& t, core::seat_index_t s) {
    ValueArray& v = eigen_util::reinterpret_as_array(t);
    eigen_util::right_rotate(v, s);
  }
};

}  // namespace core
