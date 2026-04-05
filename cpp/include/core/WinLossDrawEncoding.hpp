#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinLossDrawPlayerResult.hpp"
#include "util/EigenUtil.hpp"

#include <algorithm>
#include <array>
#include <iostream>

namespace core {

/*
 * WinLossDrawEncoding encodes a 2-player win/loss/draw GameOutcome into a neural-network-compatible
 * tensor, and provides utility methods for display. Used with WinLossDrawPlayerResult.
 */
struct WinLossDrawEncoding {
  using PlayerResult = core::WinLossDrawPlayerResult;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<3>>;  // W, L, D
  using ValueArray = eigen_util::FArray<2>;

  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor encode(const std::array<PlayerResult, 2>& outcome) {
    Tensor t;
    t.setZero();
    // outcome[0] describes the result from seat-0's perspective
    t(static_cast<int>(outcome[0].kind)) = 1.0f;
    return t;
  }

  static ValueArray to_value_array(const Tensor& t) {
    ValueArray a;
    a(0) = t(0) + 0.5f * t(2);
    a(1) = t(1) + 0.5f * t(2);
    return a;
  }

  static void left_rotate(Tensor& t, core::seat_index_t s) {
    if (s) {
      std::swap(t(0), t(1));
    }
  }

  static void right_rotate(Tensor& t, core::seat_index_t s) { left_rotate(t, s); }

  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates) {
    ValueArray net_value_array;
    ValueArray net_draw_array;
    ValueArray player_array;
    for (int i = 0; i < 2; i++) {
      player_array(i) = i;
      net_value_array(i) = net_value(i);
      net_draw_array(i) = net_value(2);
    }
    return eigen_util::concatenate_columns(player_array, net_value_array, net_draw_array,
                                           win_rates);
  }

  static const std::vector<std::string>& get_column_names() {
    static const std::vector<std::string> columns = {"Player", "Net(W)", "Net(D)", "win-rate"};
    return columns;
  }

  template <typename GameIO>
  static void print_array(GameIO, const Tensor& net_value, const ValueArray& win_rates) {
    auto data = get_data_matrix(net_value, win_rates);
    const auto& columns = get_column_names();

    eigen_util::PrintArrayFormatMap fmt_map{
      {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
    };

    eigen_util::print_array(std::cout, data, columns, &fmt_map);
  }

  template <typename GameIO>
  static boost::json::object to_json(GameIO, const Tensor& net_value, const ValueArray& win_rates) {
    auto data = get_data_matrix(net_value, win_rates);
    const auto& columns = get_column_names();

    eigen_util::PrintArrayFormatMap fmt_map{
      {"Player", [&](float x, int) { return GameIO::player_to_str(x); }},
    };

    return eigen_util::output_to_json(data, columns, &fmt_map);
  }
};

}  // namespace core
