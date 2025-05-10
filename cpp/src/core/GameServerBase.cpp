#include <core/GameServerBase.hpp>

#include <core/GameServerClient.hpp>
#include <util/Asserts.hpp>
#include <util/LoggingUtil.hpp>

namespace core {

GameServerBase::server_vec_t GameServerBase::game_servers_;

void GameServerBase::add_client(GameServerClient* client) {
  util::release_assert(!game_servers_.empty());
  for (auto server : game_servers_) {
    server->clients_.push_back(client);
  }
}

void GameServerBase::force_progress() {
  LOG_DEBUG("<-- GameServerBase: forcing progress");

  for (auto client : clients_) {
    client->handle_force_progress();
  }
}

}  // namespace core
