#pragma once

#include "core/BasicTypes.hpp"
#include "core/concepts/GameResultsConcept.hpp"
#include "util/EigenUtil.hpp"

#include <algorithm>
#include <iostream>

namespace core {

/*
 * WinLossResults can be used for 2-player games with win/loss outcomes, with no draws. If draws
 * are allowed, use WinLossDrawResults instead.
 */
struct WinLossResults {
  using Tensor = eigen_util::FTensor<Eigen::Sizes<2>>;  // W, L
  using ValueArray = eigen_util::FArray<2>;

  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor win(core::seat_index_t seat) {
    Tensor result;
    result.setZero();
    result(seat) = 1;
    return result;
  }

  static ValueArray to_value_array(const Tensor& t) {
    ValueArray a;
    a(0) = t(0);
    a(1) = t(1);
    return a;
  }

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

  static void print_array(const Tensor& net_value, const ValueArray& win_rates,
                          const eigen_util::PrintArrayFormatMap* fmt_map = nullptr) {
    auto data = get_data_matrix(net_value, win_rates);
    const auto& columns = get_column_names();
    eigen_util::print_array(std::cout, data, columns, fmt_map);
  }

  static boost::json::object to_json(const Tensor& net_value, const ValueArray& win_rates,
                                    const eigen_util::PrintArrayFormatMap* fmt_map = nullptr) {
    auto data = get_data_matrix(net_value, win_rates);
    const auto& columns = get_column_names();
    return eigen_util::output_to_json(data, columns, fmt_map);
  }
};

static_assert(concepts::GameResults<WinLossResults>);

}  // namespace core
