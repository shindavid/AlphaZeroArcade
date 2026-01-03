#include "search/ManagerParamsBase.hpp"

#include "util/BoostUtil.hpp"
#include "util/Exceptions.hpp"

#include <boost/filesystem.hpp>
#include <magic_enum/magic_enum.hpp>
#include <magic_enum/magic_enum_format.hpp>

namespace search {

template <core::concepts::EvalSpec EvalSpec>
inline ManagerParamsBase<EvalSpec>::ManagerParamsBase(search::Mode m) : mode(m) {
  if (m == search::kCompetition) {
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
      .template add_flag<"enable-pondering", "disable-pondering">(
        &enable_pondering, "enable pondering (search during opponent's turn)",
        "disable pondering (search during opponent's turn)")
      .template add_hidden_option<"pondering-tree-size-limit">(
        po::value<int>(&pondering_tree_size_limit)->default_value(pondering_tree_size_limit),
        "max tree size to grow to when pondering (only respected in --enable-pondering mode)");

  return out.add(NNEvaluationServiceParams::make_options_description());
}

}  // namespace search
