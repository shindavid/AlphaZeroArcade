#pragma once

#include "core/BasicTypes.hpp"
#include "core/WinLossDrawPlayerResult.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/EigenUtil.hpp"

namespace core {

/*
 * WinLossDrawEncoding encodes a 2-player win/loss/draw GameOutcome into a neural-network-compatible
 * tensor, and provides utility methods for display. Used with WinLossDrawPlayerResult.
 */
template <concepts::Game Game>
struct WinLossDrawEncoding {
  static_assert(Game::Constants::kNumPlayers == 2);
  using PlayerResult = core::WinLossDrawPlayerResult;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<3>>;  // W, L, D
  using GameOutcome = Game::Types::GameOutcome;
  using ValueArray = Game::Types::ValueArray;
  using GameIO = Game::IO;

  static constexpr float kMaxValue = 1.0;
  static constexpr float kMinValue = 0.0;

  static Tensor encode(const GameOutcome& outcome);
  static ValueArray to_value_array(const Tensor& t);
  static void left_rotate(Tensor& t, core::seat_index_t s);
  static void right_rotate(Tensor& t, core::seat_index_t s) { left_rotate(t, s); }
  static auto get_data_matrix(const Tensor& net_value, const ValueArray& win_rates);
  static const std::vector<std::string>& get_column_names();
  static void print_array(const Tensor& net_value, const ValueArray& win_rates);
  static boost::json::object to_json(const Tensor& net_value, const ValueArray& win_rates);
};

}  // namespace core

#include "inline/core/WinLossDrawEncoding.inl"
