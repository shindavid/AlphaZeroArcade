#pragma once

#include <string>
#include <vector>

#include <common/AbstractPlayerGenerator.hpp>
#include <common/BasicTypes.hpp>
#include <common/GameStateConcept.hpp>
#include <common/RemotePlayerProxy.hpp>
#include <util/SocketUtil.hpp>

namespace common {

template <GameStateConcept GameState>
class RemotePlayerProxyGenerator : public AbstractPlayerGenerator<GameState> {
 public:
  using base_t = AbstractPlayerGenerator<GameState>;

  void initialize(const std::string& name, io::Socket* socket, int max_simultaneous_games, player_id_t player_id);
  bool initialized() const { return socket_; }
  io::Socket* get_socket() const { return socket_; }

  std::vector<std::string> get_types() const override { return { "Remote" }; }
  std::string get_description() const override { return "Remote player from another process"; }
  AbstractPlayer<GameState>* generate(game_thread_id_t) override;
  int max_simultaneous_games() const override;

 private:
  io::Socket* socket_ = nullptr;
  int max_simultaneous_games_ = -1;
  player_id_t player_id_ = -1;
};

}  // namespace common

#include <common/inl/RemotePlayerProxyGenerator.inl>
