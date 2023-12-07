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
  using GameServerProxy = core::GameServerProxy<GameState>;
  using Player = core::AbstractPlayer<GameState>;

  struct Args {
    std::string cmd_server_hostname = "localhost";
    io::port_t cmd_server_port = 0;
    std::vector<std::string> player_strs;

    auto make_options_description() {
      namespace po = boost::program_options;
      namespace po2 = boost_util::program_options;

      po2::options_description desc("Program options");

      return desc
          .template add_option<"cmd-server-hostname">(
              po::value<std::string>(&cmd_server_hostname)->default_value(cmd_server_hostname),
              "cmd server hostname")
          .template add_option<"cmd-server-port">(
              po::value<io::port_t>(&cmd_server_port)->default_value(cmd_server_port),
              "cmd server port. If unset, then this runs without a cmd server")
          .template add_option<"player">(po::value<std::vector<std::string>>(&player_strs),
                                         "Space-delimited list of player options, wrapped "
                                         "in quotes, to be specified multiple times");
    }
  };

  static GameServer::Params get_default_game_server_params() {
    typename GameServer::Params parallel_game_runner_params;
    parallel_game_runner_params.display_progress_bar = true;
    return parallel_game_runner_params;
  }

  static int main(int ac, char* av[]) {
    try {
      namespace po = boost::program_options;
      namespace po2 = boost_util::program_options;

      Args args;
      typename GameServerProxy::Params game_server_proxy_params;
      typename GameServer::Params game_server_params = get_default_game_server_params();

      po2::options_description raw_desc("General options");
      auto desc = raw_desc.template add_option<"help", 'h'>("help")
                      .template add_option<"help-full">("help with no-op flags included")
                      .add(args.make_options_description())
                      .add(game_server_params.make_options_description())
                      .add(game_server_proxy_params.make_options_description());

      po::variables_map vm = po2::parse_args(desc, ac, av);

      PlayerFactory player_factory;
      bool help_full = vm.count("help-full");
      bool help = vm.count("help");
      if (help || help_full) {
        po2::Settings::help_full = help_full;
        std::cout << desc << std::endl;
        player_factory.print_help(args.player_strs);
        return 0;
      }

      if (args.cmd_server_port > 0) {
        core::CmdServerClient::init(args.cmd_server_hostname, args.cmd_server_port);
      }

      if (game_server_proxy_params.remote_port) {
        GameServerProxy proxy(game_server_proxy_params);

        for (const auto& pgs : player_factory.parse(args.player_strs)) {
          proxy.register_player(pgs.seat, pgs.generator);
        }
        proxy.run();
      } else {
        GameServer server(game_server_params);

        for (const auto& pgs : player_factory.parse(args.player_strs)) {
          server.register_player(pgs.seat, pgs.generator);
        }
        server.run();
      }
    } catch (const util::CleanException& e) {
      std::cerr << "Caught a CleanException: ";
      std::cerr << e.what() << std::endl;
      return 1;
    }

    return 0;
  }
};