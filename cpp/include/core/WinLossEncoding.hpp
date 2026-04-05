#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinLossPlayerResult.hpp"
#include "util/EigenUtil.hpp"

#include <algorithm>
#include <array>
#include <iostream>

namespace core {

/*
 * WinLossEncoding encodes a 2-player win/loss GameOutcome into a neural-network-compatible tensor,
 * and provides utility methods for display. Used with WinLossPlayerResult.
 */
struct WinLossEncoding {
  using PlayerResult = core::WinLossPlayerResult;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<2>>;
  using ValueArray = eigen_util::FArray<2>;

  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor encode(const std::array<PlayerResult, 2>& outcome) {
    Tensor t;
    t.setZero();
    t(0) = (outcome[0].kind == PlayerResult::kWin) ? 1.0f : 0.0f;
    t(1) = (outcome[1].kind == PlayerResult::kWin) ? 1.0f : 0.0f;
    return t;
  }

  static ValueArray to_value_array(const Tensor& t) { return eigen_util::reinterpret_as_array(t); }

  static void left_rotate(Tensor& t, core::seat_index_t) { std::swap(t(0), t(1)); }

  static void right_rotate(Tensor& t, core::seat_index_t) { std::swap(t(0), t(1)); }

  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates) {
    ValueArray player_array;
    for (int i = 0; i < 2; i++) {
      player_array(i) = i;
    }
    return eigen_util::concatenate_columns(player_array, win_rates);
  }

  static const std::vector<std::string>& get_column_names() {
    static const std::vector<std::string> columns = {"Player", "win-rate"};
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
