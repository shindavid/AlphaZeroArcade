/*
 * TODO: replace all hard-coded c4 classes/functions here with macro values. Then, have the build.py process accept
 * a python file that specifies the game name, along with other relevant metadata (like filesystem paths to the relevant
 * c++ code). The build.py process can then import that file, and pass the macro values to cmake, ultimately causing
 * this file to get compiled to a game-specific binary.
 */
#include <array>
#include <csignal>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <common/GameRunner.hpp>
#include <common/MctsPlayer.hpp>
#include <common/ParallelGameRunner.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/OracleGradedMctsPlayer.hpp>
#include <util/ParamDumper.hpp>
#include <util/StringUtil.hpp>

namespace po = boost::program_options;
namespace po2 = boost_util::program_options;

struct Args {
  std::string nnet_filename2;
  bool grade_moves = false;

  auto make_options_description() {
    po2::options_description desc("CompetitiveSelfPlay options");

    return desc
        .template add_option<"nnet-filename2">(po::value<std::string>(&nnet_filename2)->default_value(""),
            "set this to use a different nnet for the second player")
        .template add_option<"grade-moves">(po::bool_switch(&grade_moves),
            "use perfect oracle to report % of moves that were correct")
        ;
  }
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;

template<typename PlayerT, typename... Ts>
PlayerT* create_player(int index, Ts&&... ts) {
  PlayerT* player = new PlayerT(std::forward<Ts>(ts)...);
  player->set_name(util::create_string("MCTS-%d", index));
  return player;
}

template<typename PlayerT, typename... Ts>
player_array_t create_players(
    const Args& args, const Mcts::Params& mcts_params, Ts&&... ts)
{
  PlayerT* p1 = create_player<PlayerT>(1, std::forward<Ts>(ts)..., mcts_params);
  PlayerT* p2;
  if (!args.nnet_filename2.empty()) {
    Mcts::Params mcts_params2(mcts_params);
    mcts_params2.nnet_filename = args.nnet_filename2;
    p2 = create_player<PlayerT>(2, std::forward<Ts>(ts)..., mcts_params2);
  } else {
    p2 = create_player<PlayerT>(2, std::forward<Ts>(ts)..., p1->get_mcts());
  }
  return player_array_t{p1, p2};;
}

ParallelGameRunner::Params get_default_parallel_game_runner_params() {
  ParallelGameRunner::Params parallel_game_runner_params;
  parallel_game_runner_params.randomize_player_order = false;
  parallel_game_runner_params.display_progress_bar = true;
  return parallel_game_runner_params;
}

int main(int ac, char* av[]) {
  Mcts::Params mcts_params(Mcts::kCompetitive);
  c4::PerfectPlayParams perfect_play_params;
  MctsPlayer::Params mcts_player_params(MctsPlayer::kCompetitive);
  ParallelGameRunner::register_signal(SIGTERM);
  ParallelGameRunner::Params parallel_game_runner_params = get_default_parallel_game_runner_params();
  Args args;

  po2::options_description raw_desc("General options");
  auto desc = raw_desc.template add_option<"help", 'h'>("help")
      .add(perfect_play_params.make_options_description())
      .add(mcts_params.make_options_description())
      .add(mcts_player_params.make_options_description())
      .add(parallel_game_runner_params.make_options_description())
      .add(args.make_options_description());

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner runner(parallel_game_runner_params);
  if (args.grade_moves) {
    c4::PerfectOracle oracle(perfect_play_params);
    c4::OracleGrader grader(&oracle);
    runner.register_players([&]() {
      return create_players<c4::OracleGradedMctsPlayer>(args, mcts_params, &grader, mcts_player_params);
    });
    runner.run();
    grader.dump();
  } else {
    runner.register_players([&]() { return create_players<MctsPlayer>(args, mcts_params, mcts_player_params); });
    runner.run();
  }

  mcts_player_params.dump();
  util::ParamDumper::add("MCTS search threads", "%d", mcts_params.num_search_threads);
  util::ParamDumper::add("MCTS max batch size", "%d", mcts_params.batch_size_limit);
  util::ParamDumper::add("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());
  util::ParamDumper::add("MCTS pct virtual-loss influenced PUCT calcs", "%.2f%%", Mcts::pct_virtual_loss_influenced_puct_calcs());
  util::ParamDumper::flush();

  return 0;
}
