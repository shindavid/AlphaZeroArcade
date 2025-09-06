#include "nnet/UniformNNEvaluationService.hpp"

namespace nnet {

template <core::concepts::EvalSpec EvalSpec>
UniformNNEvaluationService<EvalSpec>::UniformNNEvaluationService() {
  this->set_init_func([](NNEvaluation* eval, const Item& item) {
    eval->uniform_init(item.node()->stable_data().valid_action_mask);
  });
}

}  // namespace nnet
