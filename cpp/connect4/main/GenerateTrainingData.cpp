#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <common/DerivedTypes.hpp>
#include <common/ParallelGameRunner.hpp>
#include <common/TrainingDataWriter.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <third_party/ProgressBar.hpp>
#include <util/BitSet.hpp>

namespace bf = boost::filesystem;

struct Args {
  int num_threads;
  int num_training_games;
  std::string c4_solver_dir_str;
  std::string games_dir_str;
};

using GameState = c4::GameState;
using Tensorizor = c4::Tensorizor;
using TrainingDataWriter = common::TrainingDataWriter<GameState, Tensorizor>;
using ParallelGameRunner = common::ParallelGameRunner<GameState>;

using GameStateTypes = common::GameStateTypes<GameState>;
using ActionMask = GameStateTypes::ActionMask;

/*
 * This interfaces with the connect4 perfect solver binary by creating a child-process that we interact with via
 * stdin/stdout, just like is done in the python version. It should be noted that an alternative is to compile a
 * perfect solver library, and to invoke its API directly, without creating a child-process. This alternative is
 * probably better. For example, we do away with the overhead of interprocess communication and string
 * construction/parsing. Also, the perfect solver has a cache under its hood, and direct usage would allow for
 * cache-sharing across the different threads (although a mutex might need to be added).
 *
 * On the other hand, this implementation was easier to write, this part of the pipeline is not a bottleneck, and
 * it is only a temporary fill-in until we get the self-play loop rolling. So, practically speaking, this is not worth
 * improving.
 */
class DataGenerator {
public:
  DataGenerator(const Args& args)
  : args_(args)
  , bar_(args.num_training_games)
  , writer_(args.games_dir_str) {}

  void launch() {
    std::vector<std::thread> threads;
    for (int i = 0; i < args_.num_threads; ++i) {
      int start = (i * args_.num_training_games / args_.num_threads);
      int end = ((i + 1) * args_.num_training_games / args_.num_threads);
      int num_games = end - start;
      threads.emplace_back([&] { run(num_games); });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

private:
  void run(int num_games) {
    c4::PerfectOracle oracle(args_.c4_solver_dir_str);

    for (int i = 0; i < num_games; ++i) {
      std::unique_lock lock(mutex_);
      bar_.update();
      lock.unlock();
      auto game_data = writer_.get_data();

      GameState state;
      Tensorizor tensorizor;
      c4::PerfectOracle::MoveHistory move_history;

      while (true) {
        auto query_result = oracle.get_best_moves(move_history);
        ActionMask best_moves = query_result.moves;
        int best_score = query_result.score;

        common::player_index_t cp = state.get_current_player();
        float cur_player_value = best_score > 0 ? +1 : (best_score < 0 ? 0 : 0.5f);

        auto slab = game_data->get_next_slab();
        auto& input = slab.input;
        auto& value = slab.value;
        auto& policy = slab.policy;

        value(cp) = cur_player_value;
        value(1 - cp) = 1 - cur_player_value;
        for (size_t k = 0; k < best_moves.size(); ++k) {
          policy.data()[k] = best_moves[k];
        }

        tensorizor.tensorize(input, state);

        ActionMask moves = state.get_valid_actions();

        int move = bitset_util::choose_random_on_index(moves);
        auto outcome = state.apply_move(move);
        tensorizor.receive_state_change(state, move);
        if (common::is_terminal_outcome(outcome)) {
          break;
        }

        move_history.append(move);
      }

      writer_.process(game_data);
    }
  }

  Args args_;
  std::mutex mutex_;
  progressbar bar_;
  TrainingDataWriter writer_;
};

int main(int ac, char* av[]) {
  Args args;

  namespace po = boost::program_options;
  po::options_description desc("Generate training data from perfect solver");
  desc.add_options()
      ("help,h", "help")
      ("num-training-games,n", po::value<int>(&args.num_training_games)->default_value(10000), "num training games")
      ("num-threads,t", po::value<int>(&args.num_threads)->default_value(8), "num threads")
      ("games-dir,g", po::value<std::string>(&args.games_dir_str)->default_value("c4_games"), "where to write games")
      ("c4-solver-dir,c", po::value<std::string>(&args.c4_solver_dir_str),
          "base dir containing c4solver bin and 7x6 book. Looks up in config.txt by default")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  ParallelGameRunner::global_params_.display_progress_bar = true;

  bf::path c4_solver_dir(args.c4_solver_dir_str);
  if (args.c4_solver_dir_str.empty()) {
    c4_solver_dir = c4::PerfectOracle::get_default_c4_solver_dir();
  }
  c4::PerfectOracle oracle(c4_solver_dir);  // create a single oracle on main thread to get clearer exceptions

  DataGenerator generator(args);
  generator.launch();
  printf("\nWrote data to: %s\n", args.games_dir_str.c_str());
  return 0;
}
