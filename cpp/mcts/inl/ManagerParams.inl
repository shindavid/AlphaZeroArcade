#include <mcts/ManagerParams.hpp>

#include <boost/filesystem.hpp>

#include <util/BoostUtil.hpp>
#include <util/Config.hpp>
#include <util/Exception.hpp>
#include <util/RepoUtil.hpp>

namespace mcts {

inline ManagerParams::ManagerParams(mcts::Mode mode) {
  if (mode == mcts::kCompetitive) {
    dirichlet_mult = 0;
    dirichlet_alpha_factor = 0;
    forced_playouts = false;
    root_softmax_temperature_str = "1";
  } else if (mode == mcts::kTraining) {
    root_softmax_temperature_str = "1.4->1.1:2*sqrt(b)";
    exploit_proven_winners = false;
  } else {
    throw util::Exception("Unknown mcts::Mode: %d", (int)mode);
  }
}

inline auto ManagerParams::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  boost::filesystem::path default_profiling_dir_path =
      util::Repo::root() / "output" / "mcts_profiling";
  std::string default_profiling_dir =
      util::Config::instance()->get("mcts_profiling_dir", default_profiling_dir_path.string());

  po2::options_description desc("Manager options");

  auto out = desc
      .template add_option<"num-search-threads", 'n'>(
          po::value<int>(&num_search_threads)->default_value(num_search_threads),
          "num search threads")
      .template add_bool_switches<"enable-pondering", "disable-pondering">(
          &enable_pondering, "enable pondering (search during opponent's turn)",
          "disable pondering (search during opponent's turn)")
      .template add_option<"pondering-tree-size-limit">(
          po::value<int>(&pondering_tree_size_limit)->default_value(pondering_tree_size_limit),
          "max tree size to grow to when pondering (only respected in --enable-pondering mode)")
      .template add_option<"root-softmax-temp">(
          po::value<std::string>(&root_softmax_temperature_str)
              ->default_value(root_softmax_temperature_str),
          "root softmax temperature")
      .template add_option<"cpuct", 'c'>(po2::float_value("%.2f", &cPUCT), "cPUCT value")
      .template add_option<"dirichlet-mult", 'd'>(po2::float_value("%.2f", &dirichlet_mult),
                                                  "dirichlet mult")
      .template add_option<"dirichlet-alpha-factor">(
          po2::float_value("%.2f", &dirichlet_alpha_factor), "dirichlet alpha factor")
      .template add_bool_switches<"forced-playouts", "no-forced-playouts">(
          &forced_playouts, "enable forced playouts", "disable forced playouts")
      .template add_bool_switches<"enable-first-play-urgency", "disable-first-play-urgency">(
          &enable_first_play_urgency, "enable first play urgency", "disable first play urgency")
#ifdef PROFILE_MCTS
      .template add_option<"profiling-dir">(
          po::value<std::string>(&profiling_dir_str)->default_value(default_profiling_dir),
          "directory in which to dump mcts profiling stats")
#endif  // PROFILE_MCTS
      ;

  return out.add(NNEvaluationServiceParams::make_options_description());
}

}  // namespace mcts
