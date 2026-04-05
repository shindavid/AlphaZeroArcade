#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinLossPlayerResult.hpp"
#include "util/EigenUtil.hpp"

#include <algorithm>
#include <array>

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

  static Tensor encode(const std::array<PlayerResult, 2>& outcome);

  static ValueArray to_value_array(const Tensor& t) { return eigen_util::reinterpret_as_array(t); }

  static void left_rotate(Tensor& t, core::seat_index_t) { std::swap(t(0), t(1)); }

  static void right_rotate(Tensor& t, core::seat_index_t) { std::swap(t(0), t(1)); }

  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates);

  static const std::vector<std::string>& get_column_names();

  template <typename GameIO>
  static void print_array(GameIO, const Tensor& net_value, const ValueArray& win_rates);

  template <typename GameIO>
  static boost::json::object to_json(GameIO, const Tensor& net_value, const ValueArray& win_rates);
};

}  // namespace core

#include "inline/core/WinLossEncoding.inl"
