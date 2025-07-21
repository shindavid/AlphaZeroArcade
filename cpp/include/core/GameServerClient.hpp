#pragma once

#include "core/GameServerBase.hpp"

namespace core {

class GameServerClient {
 public:
  GameServerClient(GameServerBase* server) { server->add_client(this); }
  virtual ~GameServerClient() = default;

  // When the GameServer has an empty queue, it will call this function to force progress.
  virtual void handle_force_progress() = 0;
};

}  // namespace core
