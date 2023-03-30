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
  using player_array_t = std::array<Player*, kNumPlayers>;

  RemotePlayerProxy(const std::string& name, int socket_descriptor);

  void start_game(game_id_t, const player_array_t& players, player_index_t seat_assignment) override;
  void receive_state_change(player_index_t, const GameState&, action_index_t, const GameOutcome&) override;
  action_index_t get_action(const GameState&, const ActionMask&) override;

private:
  const int socket_descriptor_;
};

}  // namespace common

#include <common/inl/RemotePlayerProxy.inl>
