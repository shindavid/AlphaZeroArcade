#pragma once

#include "beta0/concepts/SpecConcept.hpp"
#include "search/Constants.hpp"
#include "search/NNEvaluationServiceParams.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
class Manager : public search::NNEvaluationServiceParams {
 public:
  struct Params {
    Params() = default;
    Params(search::Mode mode);

    auto make_options_description();
    bool operator==(const Params&) const = default;
  };
};

}  // namespace beta0
