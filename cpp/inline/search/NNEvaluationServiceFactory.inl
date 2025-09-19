#include "search/NNEvaluationServiceFactory.hpp"

#include "search/NNEvaluationService.hpp"
#include "search/UniformNNEvaluationService.hpp"
#include "util/Exceptions.hpp"

namespace search {

template <search::concepts::Traits Traits>
typename NNEvaluationServiceFactory<Traits>::ServiceBase_ptr
NNEvaluationServiceFactory<Traits>::create(const NNEvaluationServiceParams& params,
                                           core::GameServerBase* server) {
  if (!params.no_model) {
    return NNEvaluationService<Traits>::create(params, server);
  } else if (params.model_filename.empty()) {
    return std::make_shared<UniformNNEvaluationService<Traits>>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }
}

}  // namespace search
