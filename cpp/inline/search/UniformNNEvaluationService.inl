#include "search/UniformNNEvaluationService.hpp"

namespace search {

template <::alpha0::concepts::Spec Spec>
UniformNNEvaluationService<Spec>::UniformNNEvaluationService() {
  this->set_init_func([](NNEvaluation* eval, const Item& item) {
    eval->uniform_init(item.node()->stable_data().num_valid_moves);
  });
}

}  // namespace search
