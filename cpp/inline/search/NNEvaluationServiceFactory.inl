#include "search/NNEvaluationServiceFactory.hpp"

#include "search/NNEvaluationService.hpp"
#include "search/UniformNNEvaluationService.hpp"
#include "util/Exceptions.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
typename NNEvaluationServiceFactory<SearchSpec>::ServiceBase_ptr
NNEvaluationServiceFactory<SearchSpec>::create(const NNEvaluationServiceParams& params,
                                               core::GameServerBase* server) {
  if (!params.no_model) {
    return NNEvaluationService<SearchSpec>::create(params, server);
  } else if (params.model_filename.empty()) {
    return std::make_shared<UniformNNEvaluationService<SearchSpec>>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }
}

}  // namespace search
