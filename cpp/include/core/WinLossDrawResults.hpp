#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameResultsConcept.hpp"
#include "util/EigenUtil.hpp"

#include <algorithm>
#include <iostream>

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

  static void right_rotate(Tensor& t, core::seat_index_t s) { left_rotate(t, s); }

  static void print_array(const Tensor& net_value, const ValueArray& win_rates,
                          const eigen_util::PrintArrayFormatMap* fmt_map = nullptr) {
    ValueArray net_value_array;
    ValueArray net_draw_array;
    ValueArray player_array;
    for (int i = 0; i < 2; i++) {
      player_array(i) = i;
      net_value_array(i) = net_value(i);
      net_draw_array(i) = net_value(2);
    }
    auto data =
      eigen_util::concatenate_columns(player_array, net_value_array, net_draw_array, win_rates);
    static std::vector<std::string> columns = {"Player", "Net(W)", "Net(D)", "win-rate"};
    eigen_util::print_array(std::cout, data, columns, fmt_map);
  }
};

static_assert(concepts::GameResults<WinLossDrawResults>);

}  // namespace core
