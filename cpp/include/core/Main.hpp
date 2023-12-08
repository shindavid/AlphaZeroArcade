#pragma once

#include <core/AbstractPlayer.hpp>
#include <core/CmdServerClient.hpp>
#include <core/GameServer.hpp>
#include <core/GameServerProxy.hpp>
#include <core/GameStateConcept.hpp>
#include <util/BoostUtil.hpp>
#include <util/Exception.hpp>
#include <util/SocketUtil.hpp>

#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <vector>

template <typename PlayerFactory>
struct Main {
  using GameState = PlayerFactory::GameState;
  using GameServer = core::GameServer<GameState>;
  using GameServerParams = typename GameServer::Params;
  using GameServerProxy = core::GameServerProxy<GameState>;
  using Player = core::AbstractPlayer<GameState>;

  struct Args {
    std::string cmd_server_hostname = "localhost";
    io::port_t cmd_server_port = 0;
    std::vector<std::string> player_strs;

    auto make_options_description();
  };

  static GameServerParams get_default_game_server_params();

  static int main(int ac, char* av[]);
};

#include <inline/core/Main.inl>
