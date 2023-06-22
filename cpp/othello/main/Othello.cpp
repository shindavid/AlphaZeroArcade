#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <core/GameServer.hpp>
#include <core/GameServerProxy.hpp>
#include <othello/GameState.hpp>
#include <othello/PlayerFactory.hpp>
#include <util/Exception.hpp>
#include <util/ParamDumper.hpp>

using GameState = othello::GameState;
using PlayerFactory = othello::PlayerFactory;

using GameServer = core::GameServer<GameState>;
using GameServerProxy = core::GameServerProxy<GameState>;
using Player = core::AbstractPlayer<GameState>;

namespace po = boost::program_options;
namespace po2 = boost_util::program_options;

struct Args {
  std::vector<std::string> player_strs;

  auto make_options_description() {
    po2::options_description desc("Othello options");

    return desc
        .template add_option<"player">(po::value<std::vector<std::string>>(&player_strs),
            "Space-delimited list of player options, wrapped in quotes, to be specified multiple times")
        ;
  }
};

GameServer::Params get_default_game_server_params() {
  GameServer::Params parallel_game_runner_params;
  parallel_game_runner_params.display_progress_bar = true;
  return parallel_game_runner_params;
}

int main(int ac, char* av[]) {
  try {
    namespace po = boost::program_options;
    namespace po2 = boost_util::program_options;

    Args args;
    GameServerProxy::Params game_server_proxy_params;
    GameServer::Params game_server_params = get_default_game_server_params();

    po2::options_description raw_desc("General options");
    auto desc = raw_desc.template add_option<"help", 'h'>("help")
        .add(args.make_options_description())
        .add(game_server_params.make_options_description())
        .add(game_server_proxy_params.make_options_description());

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
    po::notify(vm);

    PlayerFactory player_factory;
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      player_factory.print_help(args.player_strs);
      return 0;
    }

    if (game_server_proxy_params.remote_port) {
      GameServerProxy proxy(game_server_proxy_params);

      for (const auto &pgs: player_factory.parse(args.player_strs)) {
        proxy.register_player(pgs.seat, pgs.generator);
      }
      proxy.run();
    } else {
      GameServer server(game_server_params);

      for (const auto &pgs: player_factory.parse(args.player_strs)) {
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
