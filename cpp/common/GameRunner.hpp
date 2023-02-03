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
  class Listener {
  public:
    virtual ~Listener() = default;
    virtual void on_game_start(game_id_t game_id) {}
    virtual void on_game_end() {}
    virtual void on_move(player_index_t player, action_index_t action) {}
  };

  enum PlayerOrder {
    kFixedPlayerSeats,
    kRandomPlayerSeats
  };

  using GameOutcome = typename GameStateTypes<GameState>::GameOutcome;
  using Player = AbstractPlayer<GameState>;
  using player_array_t = std::array<Player*, GameState::kNumPlayers>;

  template<typename T> GameRunner(T&& players) : players_(players) {}

  GameOutcome run(PlayerOrder);

  void add_listener(Listener* listener) { listeners_.push_back(listener); }

private:
  using listener_vec_t = std::vector<Listener*>;
  listener_vec_t listeners_;
  player_array_t players_;
};

}  // namespace common

#include <common/inl/GameRunner.inl>
