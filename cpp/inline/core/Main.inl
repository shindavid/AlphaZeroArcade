#include <core/Main.hpp>
#include <util/KeyValueDumper.hpp>
#include <util/LoggingUtil.hpp>

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
    util::Logging::Params log_params;
    util::Random::Params random_params;
    core::LoopControllerClient::Params loop_controller_params;
    typename GameServerProxy::Params game_server_proxy_params;
    typename GameServer::Params game_server_params = get_default_game_server_params();
    TrainingDataWriterParams training_data_writer_params;

    po2::options_description raw_desc("General options");
    auto desc = raw_desc.template add_option<"help", 'h'>("help (most used options)")
                    .template add_option<"help-full">("help (all options)")
                    .add(args.make_options_description())
                    .add(training_data_writer_params.make_options_description())
                    .add(log_params.make_options_description())
                    .add(random_params.make_options_description())
                    .add(loop_controller_params.make_options_description())
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

    util::Logging::init(log_params);
    util::Random::init(random_params);

    LOG_INFO("Starting process {}", getpid());

    if (loop_controller_params.loop_controller_port > 0) {
      core::LoopControllerClient::init(loop_controller_params);
    }

    core::LoopControllerClient* client = core::LoopControllerClient::get();
    if (game_server_proxy_params.remote_port) {
      GameServerProxy proxy(game_server_proxy_params, game_server_params.num_game_threads);
      player_factory.set_server(&proxy);

      for (const auto& pgs : player_factory.parse(args.player_strs)) {
        proxy.register_player(pgs.seat, pgs.generator);
      }
      if (client) {
        client->start();
      }
      proxy.run();
    } else {
      GameServer server(game_server_params, training_data_writer_params);
      player_factory.set_server(&server);

      for (const auto& pgs : player_factory.parse(args.player_strs)) {
        server.register_player(pgs.seat, pgs.generator);
      }
      if (client) {
        client->start();
      }
      server.run();
    }

    util::KeyValueDumper::flush();

    if (client) {
      client->shutdown();
    }
  } catch (const util::CleanException& e) {
    std::cerr << "Caught a CleanException: ";
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
