#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>
#include <highfive/H5File.hpp>
#include <torch/torch.h>

#include <connect4/Constants.hpp>
#include <connect4/C4GameLogic.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/Exception.hpp>
#include <util/HighFiveUtil.hpp>
#include <util/ProgressBar.hpp>
#include <util/StringUtil.hpp>

void run(int thread_id, int num_games, const boost::filesystem::path& c4_solver_bin,
         const boost::filesystem::path& c4_solver_book, const boost::filesystem::path& games_dir)
{
  namespace bp = boost::process;

  std::string c4_cmd = util::create_string("%s -b %s -a", c4_solver_bin.c_str(), c4_solver_book.c_str());
  bp::ipstream out;
  bp::opstream in;
  bp::child proc(c4_cmd, bp::std_out > out, bp::std_in < in);

  std::string output_filename = util::create_string("%d.h5", thread_id);
  boost::filesystem::path output_path = games_dir / output_filename;

  HighFive::File h5file(output_path.string(), HighFive::File::ReadWrite | HighFive::File::Create);

  size_t max_rows = num_games * c4::kNumColumns * c4::kNumRows;
  hi5::shape_t shape = hi5::to_shape(max_rows, c4::Tensorizor::kShape);
  HighFive::DataSpace space(shape);
  HighFive::DataSet dataset = h5file.createDataSet<float>("input", space);

  hi5::shape_t offset = hi5::zeros_like(shape);
  hi5::shape_t count = hi5::to_shape(1, c4::Tensorizor::kShape);

  progressbar bar(num_games);
  for (int i = 0; i < num_games; ++i) {
    bar.update();

    c4::GameState state;
    c4::Tensorizor tensorizor;
    std::string move_history;

    while (true) {
      in.write((move_history + "\n").c_str(), move_history.size() + 1);
      in.flush();

      char buf[1024];
      out.read(buf, sizeof(buf));

    }

    dataset.select(offset, count).write(data);
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

  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; ++i) {
    int num_games = ((i + 1) * num_training_games / num_threads) - (i * num_training_games / num_threads);
    threads.emplace_back(run, i, num_games, c4_solver_bin, c4_solver_book, games_dir);
  }

  for (auto& th : threads) th.join();

  return 0;
}
