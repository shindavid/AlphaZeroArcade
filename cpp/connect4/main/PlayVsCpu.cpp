#include <iostream>

#include <boost/program_options.hpp>

#include <common/BasicTypes.hpp>
#include <common/GameRunner.hpp>
#include <common/HumanTuiPlayer.hpp>
#include <common/MctsPlayer.hpp>
#include <connect4/C4CheatingHumanTuiPlayer.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/Random.hpp>

namespace po = boost::program_options;
namespace po2 = boost_util::program_options;

struct Args {
  std::string my_starting_color;
  bool perfect;
  bool cheat_mode;

  auto make_options_description() {
    po2::options_description desc("PlayVsCpu options");

    return desc
        .template add_option<"my-starting-color", 's'>(po::value<std::string>(&my_starting_color),
            "human's starting color (R or Y). Default: random")
        .template add_option<"perfect", 'p'>(po::bool_switch(&perfect)->default_value(false),
            "play against perfect player")
        .template add_option<"cheat-mode", 'C'>(po::bool_switch(&cheat_mode)->default_value(false),
            "show winning moves")
        ;
  }
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;
using Mcts = common::Mcts<GameState, Tensorizor>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;

common::player_index_t parse_color(const std::string& str) {
  if (str == "R") return c4::kRed;
  if (str == "Y") return c4::kYellow;
  if (str.empty()) return util::Random::uniform_sample(0, c4::kNumPlayers);
  throw util::Exception("Invalid --my-starting-color/-s value: \"%s\"", str.c_str());
}

int main(int ac, char* av[]) {
  Mcts::Params mcts_params(Mcts::kCompetitive);
  MctsPlayer::Params mcts_player_params(MctsPlayer::kCompetitive);
  c4::PerfectPlayParams perfect_play_params;
  Args args;

  po2::options_description raw_desc("General options");
  auto desc = raw_desc.template add_option<"help", 'h'>("help")
      .add(mcts_params.make_options_description())
      .add(mcts_player_params.make_options_description())
      .add(perfect_play_params.make_options_description())
      .add(args.make_options_description());

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  c4::Player* human;
  if (args.cheat_mode) {
    mcts_player_params.verbose = true;
    human = new c4::CheatingHumanTuiPlayer(perfect_play_params);
  } else {
    using C4HumanTuiPlayer = common::HumanTuiPlayer<c4::GameState>;
    human = new C4HumanTuiPlayer();
  }

  c4::Player* cpu;
  if (args.perfect) {
    cpu = new c4::PerfectPlayer(perfect_play_params);
  } else {
    cpu = new MctsPlayer(mcts_player_params, mcts_params);
  }

  common::player_index_t my_color = parse_color(args.my_starting_color);
  common::player_index_t cpu_color = 1 - my_color;

  using player_t = common::AbstractPlayer<c4::GameState>;
  using player_array_t = std::array<player_t*, c4::kNumPlayers>;

  player_array_t players;
  players[my_color] = human;
  players[cpu_color] = cpu;

  using GameRunner = common::GameRunner<c4::GameState>;
  GameRunner runner(players);
  auto outcome = runner.run(GameRunner::kFixedPlayerSeats);

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
