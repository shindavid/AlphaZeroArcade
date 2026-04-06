#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinLossPlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/EigenUtil.hpp"

namespace core {

/*
 * WinLossEncoding encodes a 2-player win/loss GameOutcome into a neural-network-compatible tensor,
 * and provides utility methods for display. Used with WinLossPlayerResult.
 */
template <concepts::Game Game>
struct WinLossEncoding {
  static_assert(Game::Constants::kNumPlayers == 2);
  using PlayerResult = core::WinLossPlayerResult;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<2>>;
  using GameOutcome = Game::Types::GameOutcome;
  using ValueArray = Game::Types::ValueArray;
  using GameIO = Game::IO;

  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor encode(const GameOutcome& outcome);
  static ValueArray to_value_array(const Tensor& t) { return eigen_util::reinterpret_as_array(t); }
  static void left_rotate(Tensor& t, core::seat_index_t) { std::swap(t(0), t(1)); }
  static void right_rotate(Tensor& t, core::seat_index_t) { std::swap(t(0), t(1)); }
  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates);
  static const std::vector<std::string>& get_column_names();
  static void print_array(const Tensor& net_value, const ValueArray& win_rates);
  static boost::json::object to_json(const Tensor& net_value, const ValueArray& win_rates);
};

}  // namespace core

#include "inline/core/WinLossEncoding.inl"
