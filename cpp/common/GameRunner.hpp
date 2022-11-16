#pragma once

#include <array>

#include <common/AbstractPlayer.hpp>
#include <common/GameStateConcept.hpp>
#include <common/Types.hpp>

namespace common {

template<GameStateConcept GameState>
class GameRunner {
public:
  using Result = GameResult<GameState::kNumPlayers>;
  using Player = AbstractPlayer<GameState>;
  using player_array_t = std::array<Player*, GameState::kNumPlayers>;

  template<typename T> GameRunner(T&& players) : players_(players) {}

  Result run();

private:
  player_array_t players_;
};

}  // namespace common

#include <common/inl/GameRunner.inl>
