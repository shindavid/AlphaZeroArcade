#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinSharePlayerResult.hpp"
#include "util/EigenUtil.hpp"

#include <array>

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

  static Tensor encode(const std::array<PlayerResult, N>& outcome);

  static ValueArray to_value_array(const Tensor& t) { return eigen_util::reinterpret_as_array(t); }

  static void left_rotate(Tensor& t, core::seat_index_t s);

  static void right_rotate(Tensor& t, core::seat_index_t s);

  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates);

  static const std::vector<std::string>& get_column_names();

  template <typename GameIO>
  static void print_array(GameIO, const Tensor& net_value, const ValueArray& win_rates);

  template <typename GameIO>
  static boost::json::object to_json(GameIO, const Tensor& net_value, const ValueArray& win_rates);
};

}  // namespace core

#include "inline/core/WinShareEncoding.inl"
