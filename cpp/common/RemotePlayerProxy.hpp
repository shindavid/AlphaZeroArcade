#pragma once

#include <common/AbstractPlayer.hpp>
#include <common/BasicTypes.hpp>
#include <common/DerivedTypes.hpp>
#include <common/GameStateConcept.hpp>

#include <string>

namespace common {

/*
 * In a server-client setup, the server process will create a RemotePlayerProxy to act as a proxy for remote
 * players. The RemotePlayerProxy will communicate with the remote player over a socket.
 */
template<GameStateConcept GameState>
class RemotePlayerProxy : public AbstractPlayer<GameState> {
public:
  static constexpr int kNumPlayers = GameState::kNumPlayers;
  using GameStateTypes = common::GameStateTypes<GameState>;
  using ActionMask = typename GameStateTypes::ActionMask;
  using GameOutcome = typename GameStateTypes::GameOutcome;
  using Player = AbstractPlayer<GameState>;

  RemotePlayerProxy(int socket_descriptor, player_id_t player_id, game_thread_id_t game_thread_id,
                    int max_simultaneous_games);

  void start_game() override;
  void receive_state_change(seat_index_t, const GameState&, action_index_t) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;
  void end_game(const GameState&, const GameOutcome&) override;
  int max_simultaneous_games() const override { return max_simultaneous_games_; }

private:
  const int socket_descriptor_;
  const player_id_t player_id_;
  const game_thread_id_t game_thread_id_;
  const int max_simultaneous_games_;
};

}  // namespace common

#include <common/inl/RemotePlayerProxy.inl>
