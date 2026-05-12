#include "core/GameServerBase.hpp"

#include "core/GameServerClient.hpp"
#include "core/TrainingParams.hpp"

namespace core {

GameServerBase::GameServerBase(int num_game_threads) : num_game_threads_(num_game_threads) {
  core::TrainingParams::instance().num_game_threads = num_game_threads;
}

void GameServerBase::add_client(GameServerClient* client) { clients_.push_back(client); }

}  // namespace core
