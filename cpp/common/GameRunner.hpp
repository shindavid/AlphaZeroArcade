#pragma once

#include <array>

#include <common/AbstractPlayer.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/BasicTypes.hpp>

namespace common {

template<GameStateConcept GameState>
class GameRunner {
public:
  using GameResult = typename GameStateTypes_<GameState>::GameResult;
  using Player = AbstractPlayer<GameState>;
  using player_array_t = std::array<Player*, GameState::kNumPlayers>;

  template<typename T> GameRunner(T&& players) : players_(players) {}

  GameResult run();

private:
  player_array_t players_;
};

}  // namespace common

#include <common/inl/GameRunner.inl>
