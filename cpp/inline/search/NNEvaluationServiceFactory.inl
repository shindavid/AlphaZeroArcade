#include "search/NNEvaluationServiceFactory.hpp"

#include "search/NNEvaluationService.hpp"
#include "search/UniformNNEvaluationService.hpp"
#include "util/Exceptions.hpp"

namespace search {

template <search::concepts::NNEvalTraits Traits>
typename NNEvaluationServiceFactory<Traits>::ServiceBase_ptr
NNEvaluationServiceFactory<Traits>::create(const NNEvaluationServiceParams& params,
                                           core::GameServerBase* server, AuxFactory aux_factory) {
  if (!params.no_model) {
    return NNEvaluationService<Traits>::create(params, server, std::move(aux_factory));
  } else if (params.model_filename.empty()) {
    auto svc = std::make_shared<UniformNNEvaluationService<Traits>>();
    // Even though the uniform service never loads main-net weights, beta0 still expects an
    // aux evaluator to exist on the service (Manager dynamic_casts it). The aux remains
    // unloaded, so its ready() returns false and any override branches in the search are
    // skipped — consistent with the "no model" semantics.
    if (aux_factory) {
      svc->set_aux_service(aux_factory());
    }
    return svc;
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }
}

}  // namespace search
