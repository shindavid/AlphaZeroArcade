#include "core/Main.hpp"

#include "util/LoggingUtil.hpp"

#include <csignal>
#include <cstdlib>
#include <unistd.h>

#ifdef MIT_TEST_MODE
static_assert(false, "MIT_TEST_MODE macro must not be defined for game-exe's");
#endif

namespace detail {

inline void cleanup_and_die(int signum = 0) {
  // Send SIGTERM to everyone in our process group (PGID = our PID)
  killpg(0, SIGTERM);

  // If this was a real signal, re-raise it so core dumps still work:
  if (signum > 0) {
    signal(signum, SIG_DFL);
    raise(signum);
  }
  std::exit(signum);
}

inline void register_handlers() {
  // Normal exit()
  std::atexit([](){ cleanup_and_die(0); });

  // C++ terminate() (uncaught exception)
  std::set_terminate([](){
    cleanup_and_die(0);
  });

  // Catch SIGINT, SIGTERM (Ctrl-C, `kill`), SIGABRT
  for (int sig : {SIGINT, SIGTERM, SIGABRT, SIGSEGV}) {
    struct sigaction sa{};
    sa.sa_handler = [](int s){ cleanup_and_die(s); };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(sig, &sa, nullptr);
  }
}

}  // namespace detail

template <typename PlayerFactory>
auto Main<PlayerFactory>::Args::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Program options");

  return desc.template add_option<"player">(po::value<std::vector<std::string>>(&player_strs),
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
    // Make sure any spawned processes are in the same process group as this one.
    if (setpgid(0, 0) < 0) perror("setpgid");
    detail::register_handlers();

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
      server.print_summary();
    }

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
