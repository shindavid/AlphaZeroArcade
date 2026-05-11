#pragma once

#include "core/GameServerBase.hpp"

namespace core {

class GameServerClient {
 public:
  GameServerClient(GameServerBase* server) { server->add_client(this); }
  virtual ~GameServerClient() = default;
};

}  // namespace core
