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

#include <common/DataExportingMctsPlayer.hpp>
#include <common/GameRunner.hpp>
#include <common/MctsPlayer.hpp>
#include <common/ParallelGameRunner.hpp>
#include <common/TrainingDataWriter.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/BoostUtil.hpp>
#include <util/StringUtil.hpp>

namespace po = boost::program_options;
namespace po2 = boost_util::program_options;

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;

using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;
using ParallelGameRunner = common::ParallelGameRunner<GameState>;
using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
using DataExportingMctsPlayer = common::DataExportingMctsPlayer<GameState, Tensorizor>;
using Mcts = common::Mcts<GameState, Tensorizor>;
using Player = common::AbstractPlayer<GameState>;
using player_array_t = Player::player_array_t;


struct Args {
  float random_start_distr_mean = 0.04 * GameState::kMaxNumLocalActions;

  auto make_options_description() {
    po2::options_description desc("TrainingSelfPlay options");

    return desc.add_option<"random-start-distr-mean", 'r'>(
        po2::float_value("%.2f", &random_start_distr_mean),
        "mean of exponential distribution to select r, the number of starting moves played randomly proportionally "
        "to the raw policy distribution of the net")
        ;
  }
};

template<typename... Ts>
DataExportingMctsPlayer* create_player(int index, Ts&&... ts) {
  DataExportingMctsPlayer* player = new DataExportingMctsPlayer(std::forward<Ts>(ts)...);
  player->set_name(util::create_string("MCTS-%d", index));
  return player;
}

player_array_t create_players(
    const Args& args, TrainingDataWriter* writer, const MctsPlayer::Params& mcts_player_params,
    const Mcts::Params& mcts_params)
{
  float lambda = 1.0 / args.random_start_distr_mean;
  int r = floor(util::Random::exponential<float>(lambda));

  MctsPlayer::Params mcts_player_params_copy(mcts_player_params);
  mcts_player_params_copy.num_raw_policy_starting_moves = r;
  DataExportingMctsPlayer* p1 = create_player(1, writer, mcts_player_params_copy, mcts_params);
  DataExportingMctsPlayer* p2 = create_player(2, writer, mcts_player_params_copy, p1->get_mcts());
  return player_array_t{p1, p2};;
}

ParallelGameRunner::Params get_default_parallel_game_runner_params() {
  ParallelGameRunner::Params parallel_game_runner_params;
  parallel_game_runner_params.num_games = 10000;
  parallel_game_runner_params.randomize_player_order = false;
  parallel_game_runner_params.display_progress_bar = false;
  return parallel_game_runner_params;
}

int main(int ac, char* av[]) {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  Mcts::Params mcts_params;
  MctsPlayer::Params mcts_player_params(MctsPlayer::kTraining);
  ParallelGameRunner::register_signal(SIGTERM);
  ParallelGameRunner::Params parallel_game_runner_params = get_default_parallel_game_runner_params();
  TrainingDataWriter::Params training_data_writer_params;
  Args args;

  po2::options_description raw_desc("General options");
  auto desc = raw_desc.add_option<"help", 'h'>("help")
      .add(mcts_params.make_options_description())
      .add(mcts_player_params.make_options_description())
      .add(parallel_game_runner_params.make_options_description())
      .add(training_data_writer_params.make_options_description())
      .add(args.make_options_description());

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner runner(parallel_game_runner_params);
  TrainingDataWriter writer(training_data_writer_params);
  runner.register_players([&]() { return create_players(args, &writer, mcts_player_params, mcts_params); });
  runner.run();

  mcts_player_params.dump();
  PARAM_DUMP("MCTS search threads", "%d", mcts_params.num_search_threads);
  PARAM_DUMP("MCTS max batch size", "%d", mcts_params.batch_size_limit);
  PARAM_DUMP("MCTS avg batch size", "%.2f", Mcts::global_avg_batch_size());

  return 0;
}
