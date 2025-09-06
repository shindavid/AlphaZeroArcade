#include "nnet/NNEvaluationServiceFactory.hpp"

#include "nnet/NNEvaluationService.hpp"
#include "nnet/UniformNNEvaluationService.hpp"
#include "util/Exceptions.hpp"

namespace nnet {

template <core::concepts::EvalSpec EvalSpec>
typename NNEvaluationServiceFactory<EvalSpec>::ServiceBase_ptr
NNEvaluationServiceFactory<EvalSpec>::create(const NNEvaluationServiceParams& params,
                                             core::GameServerBase* server) {
  if (!params.no_model) {
    return NNEvaluationService<EvalSpec>::create(params, server);
  } else if (params.model_filename.empty()) {
    return std::make_shared<UniformNNEvaluationService<EvalSpec>>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }
}

}  // namespace nnet
