#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "search/Constants.hpp"
#include "search/NNEvaluationServiceParams.hpp"
#include "search/SearchParams.hpp"

#include <boost/filesystem.hpp>

namespace search {

// Base class for ManagerParams for different paradigms. For now, all our paradigms are identical,
// so everything lives in this base class. As we specialize beta0 more, we may want to move some of
// the parameters to derived classes.
template <core::concepts::EvalSpec EvalSpec>
struct ManagerParamsBase : public search::NNEvaluationServiceParams {
  ManagerParamsBase(search::Mode);

  search::SearchParams pondering_search_params() const {
    return search::SearchParams::make_pondering_params(pondering_tree_size_limit);
  }

  auto make_options_description();
  bool operator==(const ManagerParamsBase& other) const = default;

  int num_search_threads = 1;
  bool enable_pondering = false;  // pondering = think during opponent's turn
  int pondering_tree_size_limit = 4096;

  search::Mode mode;

  /*
   * If true, we forcibly evaluate all children of root nodes. This is needed in training mode to
   * create action-value targets.
   */
  bool force_evaluate_all_root_children = false;
};

}  // namespace search

#include "inline/search/ManagerParamsBase.inl"
