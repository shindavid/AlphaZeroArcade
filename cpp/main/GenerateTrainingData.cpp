#include <iostream>
#include <string>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

void run(int thread_id, int num_games, const std::string& c4_solver_dir, const std::string& games_dir) {
  std::cout << "Running " << thread_id << std::endl;
}

int main(int ac, char* av[]) {
  int num_threads;
  int num_training_games;
  std::string c4_solver_dir;
  std::string games_dir;

  namespace po = boost::program_options;
  po::options_description desc("Generate training data from perfect solver");
  desc.add_options()
      ("help,h", "product help message")
      ("num-training-games,n", po::value<int>(&num_training_games)->default_value(4), "num training games")
      ("num-threads,t", po::value<int>(&num_threads)->default_value(1), "num threads")
      ("games-dir,g", po::value<std::string>(&games_dir)->default_value("c4_games"), "where to write games")
      ("c4-solver-dir,c", po::value<std::string>(&c4_solver_dir), "base dir containing c4solver bin and 7x6 book")
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

  if (boost::filesystem::is_directory(games_dir)) {
    boost::filesystem::remove_all(games_dir);
  }
  boost::filesystem::create_directories(games_dir);

  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; ++i) {
    int num_games = ((i + 1) * num_training_games / num_threads) - (i * num_training_games / num_threads);
    threads.emplace_back(run, i, num_games, c4_solver_dir, games_dir);
  }

  for (auto& th : threads) th.join();

  return 0;
}
