#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinSharePlayerResult.hpp"
#include "util/EigenUtil.hpp"

#include <array>
#include <iostream>

namespace core {

/*
 * WinShareEncoding<N> encodes an N-player fractional win-share GameOutcome into a
 * neural-network-compatible tensor, and provides utility methods for display.
 * Used with WinSharePlayerResult.
 */
template <int N>
struct WinShareEncoding {
  using PlayerResult = core::WinSharePlayerResult;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<N>>;
  using ValueArray = eigen_util::FArray<N>;

  static constexpr int kNumPlayers = N;
  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor encode(const std::array<PlayerResult, N>& outcome) {
    ValueArray shares;
    for (int i = 0; i < N; i++) {
      shares(i) = outcome[i].share;
    }
    return eigen_util::reinterpret_as_tensor(shares);
  }

  static ValueArray to_value_array(const Tensor& t) { return eigen_util::reinterpret_as_array(t); }

  static void left_rotate(Tensor& t, core::seat_index_t s) {
    ValueArray& v = eigen_util::reinterpret_as_array(t);
    eigen_util::left_rotate(v, s);
  }

  static void right_rotate(Tensor& t, core::seat_index_t s) {
    ValueArray& v = eigen_util::reinterpret_as_array(t);
    eigen_util::right_rotate(v, s);
  }

  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates) {
    ValueArray net_value_array;
    ValueArray player_array;
    for (int i = 0; i < N; i++) {
      player_array(i) = i;
      net_value_array(i) = net_value(i);
    }
    return eigen_util::concatenate_columns(player_array, net_value_array, win_rates);
  }

  static const std::vector<std::string>& get_column_names() {
    static const std::vector<std::string> columns = {"Player", "Net(W)", "win-rate"};
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
