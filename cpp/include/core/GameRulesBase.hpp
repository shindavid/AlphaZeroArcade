#pragma once

#include "core/BasicTypes.hpp"
#include "core/RulesResult.hpp"
#include "core/TrivialChanceDistribution.hpp"
#include "core/WinLossDrawPlayerResult.hpp"
#include "core/WinLossPlayerResult.hpp"
#include "util/Exceptions.hpp"

namespace core {

template <typename Types>
struct RulesBase {
  using State = Types::State;
  using Move = Types::Move;
  using PlayerResult = Types::PlayerResult;
  using GameOutcome = Types::GameOutcome;
  using Result = core::RulesResult<Types>;
  using TrivialChanceDistribution = core::TrivialChanceDistribution<Move>;

  static constexpr int kNumPlayers = Types::kNumPlayers;

  static constexpr bool is_chance_state(const State&) { return false; }

  static TrivialChanceDistribution get_chance_distribution(const State& state) {
    throw util::Exception("Chance distribution not implemented for this game");
  }

  static void backtrack_state(State& state, const State& prev_state) { state = prev_state; }

  /*
   * Construct the GameOutcome for a resignation by the given seat. The resigning player receives
   * a loss; all other players receive a win.
   *
   * A default implementation is provided for WinLossPlayerResult and WinLossDrawPlayerResult.
   * For other PlayerResult types (e.g. WinSharePlayerResult in multiplayer games), the semantics
   * of resignation are game-specific, so this method throws by default.
   */
  static GameOutcome make_resignation(seat_index_t resigning_seat) {
    if constexpr (std::is_same_v<PlayerResult, WinLossPlayerResult> ||
                  std::is_same_v<PlayerResult, WinLossDrawPlayerResult>) {
      GameOutcome outcome;
      for (int s = 0; s < kNumPlayers; ++s) {
        outcome[s].kind = (s == resigning_seat) ? PlayerResult::kLoss : PlayerResult::kWin;
      }
      return outcome;
    } else {
      throw util::Exception("make_resignation() not supported for this game's PlayerResult type");
    }
  }
};
}  // namespace core
