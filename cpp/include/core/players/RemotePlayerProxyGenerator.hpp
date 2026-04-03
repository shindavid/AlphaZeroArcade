#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/BasicTypes.hpp"
#include "core/GameServerBase.hpp"
#include "core/concepts/GameConcept.hpp"
#include "util/SocketUtil.hpp"

#include <string>

namespace core {

template <concepts::Game Game>
class RemotePlayerProxyGenerator : public AbstractPlayerGenerator<Game> {
 public:
  using base_t = AbstractPlayerGenerator<Game>;

  RemotePlayerProxyGenerator(GameServerBase*) {}

  void initialize(io::Socket* socket, int max_simultaneous_games, player_id_t player_id);
  bool initialized() const { return socket_; }
  io::Socket* get_socket() const { return socket_; }

  std::string get_default_name() const override { return "Remote"; }
  std::string type_str() const override { return "Remote"; }
  std::string get_description() const override { return "Remote player from another process"; }
  AbstractPlayer<Game>* generate(game_slot_index_t) override;
  void end_session() override;
  int max_simultaneous_games() const override;

 private:
  io::Socket* socket_ = nullptr;
  int max_simultaneous_games_ = -1;
  player_id_t player_id_ = -1;
};

}  // namespace core

#include "inline/core/players/RemotePlayerProxyGenerator.inl"
