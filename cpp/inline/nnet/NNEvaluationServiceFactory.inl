#include "nnet/NNEvaluationServiceFactory.hpp"

#include "nnet/NNEvaluationService.hpp"
#include "nnet/UniformNNEvaluationService.hpp"
#include "util/Exceptions.hpp"

namespace nnet {

template <typename Traits>
typename NNEvaluationServiceFactory<Traits>::ServiceBase_ptr
NNEvaluationServiceFactory<Traits>::create(const ManagerParams& params,
                                           core::GameServerBase* server) {
  if (!params.no_model) {
    return NNEvaluationService<Traits>::create(params, server);
  } else if (params.model_filename.empty()) {
    return std::make_shared<UniformNNEvaluationService<Traits>>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }
}

}  // namespace nnet
