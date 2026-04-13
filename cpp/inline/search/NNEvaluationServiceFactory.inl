#include "search/NNEvaluationServiceFactory.hpp"

#include "search/NNEvaluationService.hpp"
#include "search/UniformNNEvaluationService.hpp"
#include "util/Exceptions.hpp"

namespace search {

template <::alpha0::concepts::Spec Spec>
typename NNEvaluationServiceFactory<Spec>::ServiceBase_ptr
NNEvaluationServiceFactory<Spec>::create(const NNEvaluationServiceParams& params,
                                               core::GameServerBase* server) {
  if (!params.no_model) {
    return NNEvaluationService<Spec>::create(params, server);
  } else if (params.model_filename.empty()) {
    return std::make_shared<UniformNNEvaluationService<Spec>>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }
}

}  // namespace search
