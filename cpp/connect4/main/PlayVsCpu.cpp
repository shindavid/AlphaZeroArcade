#include <iostream>

#include <boost/program_options.hpp>

#include <common/BasicTypes.hpp>
#include <common/GameRunner.hpp>
#include <common/HumanTuiPlayer.hpp>
#include <common/NNetPlayer.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/Random.hpp>

struct Args {
  std::string c4_solver_dir_str;
  std::string my_starting_color;
  int num_mcts_iters;
  int num_search_threads;
  float temperature;
  bool perfect;
  bool verbose;
};

common::player_index_t parse_color(const std::string& str) {
  if (str == "R") return c4::kRed;
  if (str == "Y") return c4::kYellow;
  if (str.empty()) return util::Random::uniform_sample(0, c4::kNumPlayers);
  throw util::Exception("Invalid --my-starting-color/-s value: \"%s\"", str.c_str());
}

int main(int ac, char* av[]) {
  Args args;

  std::string default_c4_solver_dir_str = util::Config::instance()->get("c4.solver_dir", "");

  namespace po = boost::program_options;
  po::options_description desc("Generate training data from perfect solver");
  desc.add_options()
      ("help,h", "help")
      ("my-starting-color,c", po::value<std::string>(&args.my_starting_color), "human's starting color (R or Y). Default: random")
      ("c4-solver-dir,d", po::value<std::string>(&args.c4_solver_dir_str)->default_value(default_c4_solver_dir_str), "base dir containing c4solver bin and 7x6 book")
      ("perfect,p", po::bool_switch(&args.perfect)->default_value(false), "play against perfect player")
      ("num-mcts-iters,m", po::value<int>(&args.num_mcts_iters)->default_value(100), "num mcts iterations to do per move")
      ("num-search-threads,s", po::value<int>(&args.num_search_threads)->default_value(8), "num mcts search threads")
      ("temperature,t", po::value<float>(&args.temperature)->default_value(0.0), "temperature. Must be >=0. Higher=more random play")
      ("verbose,v", po::bool_switch(&args.verbose)->default_value(false), "verbose mode")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (args.c4_solver_dir_str.empty()) {
    throw util::Exception("Must either pass -c or add an entry for \"c4.solver_dir\" to %s",
                          util::Config::instance()->config_path().c_str());
  }

  boost::filesystem::path c4_solver_dir(args.c4_solver_dir_str);
  using C4HumanTuiPlayer = common::HumanTuiPlayer<c4::GameState>;
  c4::Player* human = new C4HumanTuiPlayer();

  c4::Player* cpu;
  if (args.perfect) {
    c4::PerfectPlayer::Params cpu_params;
    cpu_params.c4_solver_dir = c4_solver_dir;
    cpu = new c4::PerfectPlayer(cpu_params);
  } else {
    using C4NNetPlayer = common::NNetPlayer<c4::GameState, c4::Tensorizor>;
    C4NNetPlayer::Params cpu_params;
    cpu_params.num_mcts_iters = args.num_mcts_iters;
    cpu_params.num_search_threads = args.num_search_threads;
    cpu_params.temperature = args.temperature;
    cpu_params.verbose = args.verbose;
    cpu = new C4NNetPlayer(cpu_params);
  }

  common::player_index_t my_color = parse_color(args.my_starting_color);
  common::player_index_t cpu_color = 1 - my_color;

  using player_t = common::AbstractPlayer<c4::GameState>;
  using player_array_t = std::array<player_t*, c4::kNumPlayers>;

  player_array_t players;
  players[my_color] = human;
  players[cpu_color] = cpu;

  common::GameRunner<c4::GameState> runner(players);
  auto outcome = runner.run();

  if (outcome[my_color] == 1) {
    std::cout << "Congratulations, you win!" << std::endl;
  } else if (outcome[cpu_color] == 1) {
    std::cout << "Sorry! You lose!" << std::endl;
  } else {
    std::cout << "The game has ended in a draw!" << std::endl;
  }

  delete cpu;
  delete human;

  return 0;
}
