#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <torch/torch.h>

#include <common/DerivedTypes.hpp>
#include <connect4/C4Constants.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4PerfectPlayer.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <third_party/ProgressBar.hpp>
#include <util/EigenTorch.hpp>
#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

namespace bf = boost::filesystem;

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
void run(int thread_id, int num_games, const bf::path& c4_solver_dir, const bf::path& games_dir)
{
  c4::PerfectOracle oracle(c4_solver_dir);
  std::string output_filename = util::create_string("%d.pt", thread_id);
  bf::path output_path = games_dir / output_filename;

  size_t max_rows = num_games * c4::kNumColumns * c4::kNumRows;

  using TensorizorTypes = common::TensorizorTypes_<c4::Tensorizor>;
  using GameStateTypes = common::GameStateTypes_<c4::GameState>;
  using ActionMask = GameStateTypes::ActionMask;

  using FullEigenTorchInput = TensorizorTypes::DynamicInputTensor;
  using FullEigenTorchValue = GameStateTypes::ValueArray<Eigen::Dynamic>;
  using FullEigenTorchPolicy = GameStateTypes::PolicyArray<Eigen::Dynamic>;

  auto full_input_shape = util::to_std_array<int>(max_rows, util::std_array_v<int, c4::Tensorizor::Shape>);
  auto full_value_shape = util::to_std_array<int>(max_rows, c4::kNumPlayers);
  auto full_policy_shape = util::to_std_array<int>(max_rows, c4::kNumColumns);

  FullEigenTorchInput full_input(full_input_shape);
  FullEigenTorchValue full_value(max_rows, c4::kNumPlayers, full_value_shape);
  FullEigenTorchPolicy full_policy(max_rows, c4::kNumColumns, full_policy_shape);

  bool use_progress_bar = thread_id == 0;
  int row = 0;
  progressbar* bar = use_progress_bar ? new progressbar(num_games) : nullptr;
  for (int i = 0; i < num_games; ++i) {
    if (use_progress_bar) bar->update();

    c4::GameState state;
    c4::Tensorizor tensorizor;
    c4::PerfectOracle::MoveHistory move_history;

    while (true) {
      auto query_result = oracle.get_best_moves(move_history);
      ActionMask best_moves = query_result.moves;
      int best_score = query_result.score;

      common::player_index_t cp = state.get_current_player();
      float cur_player_value = best_score > 0 ? +1 : (best_score < 0 ? 0 : 0.5f);

      auto& input = full_input.eigenSlab<TensorizorTypes::Shape>(row);
      auto& value = full_value.eigenSlab(row);
      auto& policy = full_policy.eigenSlab(row);

      value(cp) = cur_player_value;
      value(1 - cp) = 1 - cur_player_value;
      best_moves.to_array(policy.data());

      tensorizor.tensorize(input, state);
      ++row;

      ActionMask moves = state.get_valid_actions();
      int move = moves.choose_random_set_bit();
      auto outcome = state.apply_move(move);
      tensorizor.receive_state_change(state, move);
      if (common::is_terminal_outcome(outcome)) {
        break;
      }

      move_history.append(move);
    }

    auto slice = torch::indexing::Slice(torch::indexing::None, row);
    using tensor_map_t = std::map<std::string, torch::Tensor>;
    tensor_map_t tensor_map;
    tensor_map["input"] = full_input.asTorch().index({slice});
    tensor_map["value"] = full_value.asTorch().index({slice});
    tensor_map["policy"] = full_policy.asTorch().index({slice});
    torch_util::save(tensor_map, output_path.string());
  }
  if (use_progress_bar) {
    delete bar;
  }
}

struct Args {
  int num_threads;
  int num_training_games;
  std::string c4_solver_dir_str;
  std::string games_dir_str;
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

  bf::path c4_solver_dir(args.c4_solver_dir_str);
  if (args.c4_solver_dir_str.empty()) {
    c4_solver_dir = c4::PerfectOracle::get_default_c4_solver_dir();
  }
  c4::PerfectOracle oracle(c4_solver_dir);  // create a single oracle on main thread to get clearer exceptions

  bf::path games_dir(args.games_dir_str);
  if (bf::is_directory(games_dir)) {
    bf::remove_all(games_dir);
  }
  bf::create_directories(games_dir);

  if (args.num_threads == 1) {  // specialize for easier to parse core-dumps
    run(0, args.num_training_games, c4_solver_dir, games_dir);
  } else {
    std::vector<std::thread> threads;
    for (int i=0; i<args.num_threads; ++i) {
      int start = (i * args.num_training_games / args.num_threads);
      int end = ((i + 1) * args.num_training_games / args.num_threads);
      int num_games = end - start;
      threads.emplace_back(run, i, num_games, c4_solver_dir, games_dir);
    }

    for (auto& th : threads) th.join();
  }
  printf("\nWrote data to: %s\n", games_dir.c_str());
  return 0;
}
