#include "search/ManagerParamsBase.hpp"

#include "util/BoostUtil.hpp"
#include "util/Exceptions.hpp"

#include <boost/filesystem.hpp>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

namespace search {

template <core::concepts::EvalSpec EvalSpec>
inline ManagerParamsBase<EvalSpec>::ManagerParamsBase(search::Mode mode) {
  if (mode == search::kCompetition) {
    dirichlet_mult = 0;
    dirichlet_alpha_factor = 0;
    forced_playouts = false;
    starting_root_softmax_temperature = 1;
    ending_root_softmax_temperature = 1;
    root_softmax_temperature_half_life = 1;
  } else if (mode == search::kTraining) {
    force_evaluate_all_root_children = true;
  } else {
    throw util::Exception("Unknown search::Mode: {}", mode);
  }
}

template <core::concepts::EvalSpec EvalSpec>
inline auto ManagerParamsBase<EvalSpec>::make_options_description() {
  namespace po = boost::program_options;
  namespace po2 = boost_util::program_options;

  po2::options_description desc("Manager options");

  auto out =
    desc
      .template add_option<"num-search-threads", 'n'>(
        po::value<int>(&num_search_threads)->default_value(num_search_threads),
        "num search threads")
      .template add_option<"cpuct", 'c'>(po2::default_value("{:.2f}", &cPUCT), "cPUCT value")
      .template add_option<"dirichlet-mult", 'd'>(po2::default_value("{:.2f}", &dirichlet_mult),
                                                  "dirichlet mult")
      .template add_hidden_option<"starting-root-softmax-temp">(
        po::value<float>(&starting_root_softmax_temperature)
          ->default_value(starting_root_softmax_temperature),
        "starting root softmax temperature")
      .template add_hidden_option<"ending-root-softmax-temp">(
        po::value<float>(&ending_root_softmax_temperature)
          ->default_value(ending_root_softmax_temperature),
        "ending root softmax temperature")
      .template add_hidden_option<"root-softmax-temp-half-life">(
        po::value<float>(&root_softmax_temperature_half_life)
          ->default_value(root_softmax_temperature_half_life),
        "root softmax temperature half-life")
      .template add_option<"dirichlet-alpha-factor">(
        po2::default_value("{:.2f}", &dirichlet_alpha_factor), "dirichlet alpha factor")
      .template add_flag<"enable-pondering", "disable-pondering">(
        &enable_pondering, "enable pondering (search during opponent's turn)",
        "disable pondering (search during opponent's turn)")
      .template add_hidden_option<"pondering-tree-size-limit">(
        po::value<int>(&pondering_tree_size_limit)->default_value(pondering_tree_size_limit),
        "max tree size to grow to when pondering (only respected in --enable-pondering mode)")
      .template add_hidden_flag<"forced-playouts", "no-forced-playouts">(
        &forced_playouts, "enable forced playouts", "disable forced playouts")
      .template add_hidden_flag<"enable-first-play-urgency", "disable-first-play-urgency">(
        &enable_first_play_urgency, "enable first play urgency", "disable first play urgency");

  return out.add(NNEvaluationServiceParams::make_options_description());
}

}  // namespace search
