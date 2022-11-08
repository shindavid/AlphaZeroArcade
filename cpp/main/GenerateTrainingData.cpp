#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>
#include <torch/torch.h>

#include <connect4/Constants.hpp>
#include <connect4/C4GameLogic.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/Exception.hpp>
#include <util/ProgressBar.hpp>
#include <util/StringUtil.hpp>
#include <util/TorchUtil.hpp>

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
void run(int thread_id, int num_games, const boost::filesystem::path& c4_solver_bin,
         const boost::filesystem::path& c4_solver_book, const boost::filesystem::path& games_dir)
{
  namespace bp = boost::process;

  std::string c4_cmd = util::create_string("%s -b %s -a", c4_solver_bin.c_str(), c4_solver_book.c_str());
  bp::ipstream out;
  bp::opstream in;
  bp::child proc(c4_cmd, bp::std_out > out,
                 bp::std_err > bp::null,
                 bp::std_in < in);

  std::string output_filename = util::create_string("%d.pt", thread_id);
  boost::filesystem::path output_path = games_dir / output_filename;

  size_t max_rows = num_games * c4::kNumColumns * c4::kNumRows;
  torch::Tensor input_tensor = torch::zeros(torch_util::to_shape(max_rows, c4::Tensorizor::kShape));
  torch::Tensor value_tensor = torch::zeros(torch_util::to_shape(max_rows, c4::kNumPlayers));
  torch::Tensor policy_tensor = torch::zeros(torch_util::to_shape(max_rows, c4::kNumColumns));

  bool use_progress_bar = thread_id == 0;
  int row = 0;
  progressbar* bar = use_progress_bar ? new progressbar(num_games) : nullptr;
  for (int i = 0; i < num_games; ++i) {
    if (use_progress_bar) bar->update();

    c4::GameState state;
    c4::Tensorizor tensorizor;
    char move_history[64];
    int mh = 0;

    while (true) {
      move_history[mh] = '\n';
      in.write(move_history, mh + 1);
      in.flush();

      std::string s;
      std::getline(out, s);
      auto tokens = util::split(s);
      int move_scores[c4::kNumColumns];
      for (int j = 0; j < c4::kNumColumns; ++j) {
        int score = std::atoi(tokens[tokens.size() - c4::kNumColumns + j].c_str());
        move_scores[j] = score;
      }

      common::player_index_t cp = state.get_current_player();
      int best_score = *std::max_element(move_scores, move_scores + c4::kNumColumns);

      float cur_player_value = best_score > 0 ? +1 : (best_score < 0 ? 0 : 0.5f);

      float value_arr[c4::kNumPlayers] = {};
      float best_move_arr[c4::kNumColumns] = {};

      for (int j = 0; j < c4::kNumColumns; ++j) {
        int best = move_scores[j] == best_score;
        best_move_arr[j] = best;
      }

      value_arr[cp] = cur_player_value;
      value_arr[1 - cp] = 1 - cur_player_value;

      tensorizor.tensorize(input_tensor.index({row}), state);
      torch_util::copy_to(value_tensor.index({row}), value_arr, c4::kNumPlayers);
      torch_util::copy_to(policy_tensor.index({row}), best_move_arr, c4::kNumColumns);
      ++row;

      c4::ActionMask moves = state.get_valid_actions();
      int move = moves.choose_random_set_bit();
      auto result = state.apply_move(move);
      tensorizor.receive_state_change(state, move);
      if (result.is_terminal()) {
//        printf("Game terminated! %s\n", state.compact_repr().c_str());
        break;
      }

      mh += sprintf(&move_history[mh], "%d", move + 1);
    }

    auto slice = torch::indexing::Slice(torch::indexing::None, row);
    using tensor_map_t = std::map<std::string, torch::Tensor>;
    tensor_map_t tensor_map;
    tensor_map["input"] = input_tensor.index({slice});
    tensor_map["value"] = value_tensor.index({slice});
    tensor_map["policy"] = policy_tensor.index({slice});
    torch_util::save(tensor_map, output_path.string());
  }
  if (use_progress_bar) {
    delete bar;
  }
}

int main(int ac, char* av[]) {
  int num_threads;
  int num_training_games;
  std::string c4_solver_dir_str;
  std::string games_dir_str;

  namespace po = boost::program_options;
  po::options_description desc("Generate training data from perfect solver");
  desc.add_options()
      ("help,h", "product help message")
      ("num-training-games,n", po::value<int>(&num_training_games)->default_value(100), "num training games")
      ("num-threads,t", po::value<int>(&num_threads)->default_value(4), "num threads")
      ("games-dir,g", po::value<std::string>(&games_dir_str)->default_value("c4_games"), "where to write games")
      ("c4-solver-dir,c", po::value<std::string>(&c4_solver_dir_str), "base dir containing c4solver bin and 7x6 book")
      ;

  po::variables_map vm;
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  if (!num_training_games) {
    throw std::runtime_error("Required option: -n");
  }

  boost::filesystem::path c4_solver_dir(c4_solver_dir_str);
  if (!boost::filesystem::is_directory(c4_solver_dir)) {
    throw util::Exception("Directory does not exist: %s", c4_solver_dir.c_str());
  }
  boost::filesystem::path c4_solver_bin = c4_solver_dir / "c4solver";
  boost::filesystem::path c4_solver_book = c4_solver_dir / "7x6.book";
  if (!boost::filesystem::is_regular_file(c4_solver_bin)) {
    throw util::Exception("File does not exist: %s", c4_solver_bin.c_str());
  }
  if (!boost::filesystem::is_regular_file(c4_solver_book)) {
    throw util::Exception("File does not exist: %s", c4_solver_book.c_str());
  }

  boost::filesystem::path games_dir(games_dir_str);
  if (boost::filesystem::is_directory(games_dir)) {
    boost::filesystem::remove_all(games_dir);
  }
  boost::filesystem::create_directories(games_dir);

  if (num_threads == 1) {  // specialize for easier to parse core-dumps
    run(0, num_training_games, c4_solver_bin, c4_solver_book, games_dir);
  } else {
    std::vector<std::thread> threads;
    for (int i=0; i<num_threads; ++i) {
      int num_games = ((i + 1) * num_training_games / num_threads) - (i * num_training_games / num_threads);
      threads.emplace_back(run, i, num_games, c4_solver_bin, c4_solver_book, games_dir);
    }

    for (auto& th : threads) th.join();
  }
  printf("\nWrote data to: %s\n", games_dir.c_str());
  return 0;
}
