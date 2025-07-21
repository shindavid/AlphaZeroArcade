#pragma once

#include "core/AbstractPlayer.hpp"
#include "core/GameServer.hpp"
#include "core/GameServerProxy.hpp"
#include "core/LoopControllerClient.hpp"
#include "core/TrainingDataWriter.hpp"
#include "core/concepts/Game.hpp"
#include "util/BoostUtil.hpp"
#include "util/Exceptions.hpp"
#include "util/SocketUtil.hpp"

#include <boost/program_options.hpp>

#include <string>
#include <vector>

template <typename PlayerFactory>
struct Main {
  using Game = PlayerFactory::Game;
  using GameServer = core::GameServer<Game>;
  using GameServerParams = GameServer::Params;
  using GameServerProxy = core::GameServerProxy<Game>;
  using Player = core::AbstractPlayer<Game>;
  using TrainingDataWriter = core::TrainingDataWriter<Game>;
  using TrainingDataWriterParams = TrainingDataWriter::Params;

  struct Args {
    std::vector<std::string> player_strs;

    auto make_options_description();
  };

  static GameServerParams get_default_game_server_params();

  static int main(int ac, char* av[]);
};

#include "inline/core/Main.inl"
