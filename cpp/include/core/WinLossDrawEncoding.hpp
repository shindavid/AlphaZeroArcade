#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinLossDrawPlayerResult.hpp"
#include "util/EigenUtil.hpp"

#include <array>

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

  static Tensor encode(const std::array<PlayerResult, 2>& outcome);

  static ValueArray to_value_array(const Tensor& t);

  static void left_rotate(Tensor& t, core::seat_index_t s);

  static void right_rotate(Tensor& t, core::seat_index_t s) { left_rotate(t, s); }

  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates);

  static const std::vector<std::string>& get_column_names();

  template <typename GameIO>
  static void print_array(GameIO, const Tensor& net_value, const ValueArray& win_rates);

  template <typename GameIO>
  static boost::json::object to_json(GameIO, const Tensor& net_value, const ValueArray& win_rates);
};

}  // namespace core

#include "inline/core/WinLossDrawEncoding.inl"
