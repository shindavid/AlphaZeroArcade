#pragma once

#include "beta0/ManagerParams.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "search/Constants.hpp"
#include "search/NNEvaluationServiceParams.hpp"

namespace beta0 {

template <beta0::concepts::Spec Spec>
class Manager : public search::NNEvaluationServiceParams {
 public:
  using Params = ManagerParams<Spec>;

  const Params& params() const { return params_; }

 private:
  const Params params_;
};

}  // namespace beta0
