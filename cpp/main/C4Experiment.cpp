#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

int main(int ac, char* av[]) {
  int num_threads;
  int num_training_games;
  std::string c4_solver_dir_str;
  std::string games_dir_str;

  namespace po = boost::program_options;
  po::options_description desc("Generate training data from perfect solver");
  desc.add_options()
      ("help,h", "product help message")
      ("num-training-games,n", po::value<int>(&num_training_games)->default_value(4), "num training games")
      ("num-threads,t", po::value<int>(&num_threads)->default_value(1), "num threads")
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
    throw std::exception();
  }
  boost::filesystem::path c4_solver_bin = c4_solver_dir / "c4solver";
  boost::filesystem::path c4_solver_book = c4_solver_dir / "7x6.book";
  if (!boost::filesystem::is_regular_file(c4_solver_bin)) {
    throw std::exception();
  }
  if (!boost::filesystem::is_regular_file(c4_solver_book)) {
    throw std::exception();
  }

  boost::filesystem::path games_dir(games_dir_str);
  if (boost::filesystem::is_directory(games_dir)) {
    boost::filesystem::remove_all(games_dir);
  }
  boost::filesystem::create_directories(games_dir);

  return 0;
}
