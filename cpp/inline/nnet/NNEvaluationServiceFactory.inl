#include "nnet/NNEvaluationServiceFactory.hpp"

#include "nnet/NNEvaluationService.hpp"
#include "nnet/UniformNNEvaluationService.hpp"
#include "util/Exceptions.hpp"

namespace nnet {

template <core::concepts::Game Game>
typename NNEvaluationServiceFactory<Game>::ServiceBase_ptr NNEvaluationServiceFactory<Game>::create(
  const NNEvaluationServiceParams& params, core::GameServerBase* server) {
  if (!params.no_model) {
    return NNEvaluationService<Game>::create(params, server);
  } else if (params.model_filename.empty()) {
    return std::make_shared<UniformNNEvaluationService<Game>>();
  } else {
    throw util::CleanException("--model_filename/-m and --no-model cannot be used together");
  }
}

}  // namespace nnet
