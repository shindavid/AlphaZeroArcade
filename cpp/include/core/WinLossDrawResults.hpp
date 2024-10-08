#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/GameResults.hpp>
#include <util/EigenUtil.hpp>

#include <algorithm>

namespace core {

/*
 * WinLossDrawResults can be used for 2-player games with win/loss/draw outcomes, where draws are
 * considered half as valuable as wins.
 */
struct WinLossDrawResults {
  using Tensor = eigen_util::FTensor<Eigen::Sizes<3>>;  // W, L, D
  using ValueArray = eigen_util::FArray<2>;

  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor draw() {
    Tensor result;
    result.setZero();
    result(2) = 1;
    return result;
  }

  static Tensor win(core::seat_index_t seat) {
    Tensor result;
    result.setZero();
    result(seat) = 1;
    return result;
  }

  static ValueArray to_value_array(const Tensor& t) {
    ValueArray a;
    a(0) = t(0) + 0.5 * t(2);
    a(1) = t(1) + 0.5 * t(2);
    return a;
  }

  static void left_rotate(Tensor& t, core::seat_index_t s) {
    if (s) {
      std::swap(t(0), t(1));
    }
  }

  static void right_rotate(Tensor& t, core::seat_index_t s) {
    left_rotate(t, s);
  }
};

static_assert(concepts::GameResults<WinLossDrawResults>);

}  // namespace core
