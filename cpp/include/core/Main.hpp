#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/LoopControllerClient.hpp>
#include <core/GameServer.hpp>
#include <core/GameServerProxy.hpp>
#include <core/concepts/Game.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/SocketUtil.hpp>

#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <vector>

template <typename PlayerFactory>
struct Main {
  using Game = typename PlayerFactory::Game;
  using GameServer = core::GameServer<Game>;
  using GameServerParams = typename GameServer::Params;
  using GameServerProxy = core::GameServerProxy<Game>;
  using Player = core::AbstractPlayer<Game>;

  struct Args {
    std::vector<std::string> player_strs;

    auto make_options_description();
  };

  static GameServerParams get_default_game_server_params();

  static int main(int ac, char* av[]);
};

#include <inline/core/Main.inl>
