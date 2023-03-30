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
  enum PlayerOrder {
    kFixedPlayerSeats,
    kRandomPlayerSeats
  };

  using GameStateTypes = common::GameStateTypes<GameState>;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using Player = AbstractPlayer<GameState>;
  using player_array_t = std::array<Player*, GameState::kNumPlayers>;
  using player_name_array_t = typename GameStateTypes::player_name_array_t;

  template<typename T> GameRunner(T&& players) : players_(players) {}

  GameOutcome run(PlayerOrder);

private:
  player_array_t players_;
};

}  // namespace common

#include <common/inl/GameRunner.inl>
