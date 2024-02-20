#include <core/Main.hpp>

template <typename PlayerFactory>
auto Main<PlayerFactory>::Args::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Program options");

  return desc
      .template add_option<"player">(po::value<std::vector<std::string>>(&player_strs),
                                     "Space-delimited list of player options, wrapped "
                                     "in quotes, to be specified multiple times");
}

template <typename PlayerFactory>
typename Main<PlayerFactory>::GameServerParams
Main<PlayerFactory>::get_default_game_server_params() {
  GameServerParams parallel_game_runner_params;
  parallel_game_runner_params.display_progress_bar = true;
  return parallel_game_runner_params;
}

template <typename PlayerFactory>
int Main<PlayerFactory>::main(int ac, char* av[]) {
  try {
    namespace po = boost::program_options;
    namespace po2 = boost_util::program_options;

    Args args;
    core::TrainingServerClient::Params training_server_params;
    typename GameServerProxy::Params game_server_proxy_params;
    typename GameServer::Params game_server_params = get_default_game_server_params();

    po2::options_description raw_desc("General options");
    auto desc = raw_desc.template add_option<"help", 'h'>("help")
                    .template add_option<"help-full">("help with no-op flags included")
                    .add(args.make_options_description())
                    .add(training_server_params.make_options_description())
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

    std::cout << util::TimestampPrefix::get() << " Starting process " << getpid() << std::endl;

    if (training_server_params.training_server_port > 0) {
      core::TrainingServerClient::init(training_server_params);
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
