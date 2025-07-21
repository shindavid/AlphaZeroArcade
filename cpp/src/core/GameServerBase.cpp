#include "core/GameServerBase.hpp"
#include "core/GameServerClient.hpp"
#include "util/Asserts.hpp"
#include "util/LoggingUtil.hpp"

namespace core {

void GameServerBase::add_client(GameServerClient* client) { clients_.push_back(client); }

void GameServerBase::force_progress() {
  LOG_DEBUG("<-- GameServerBase: forcing progress");

  for (auto client : clients_) {
    client->handle_force_progress();
  }
}

}  // namespace core
