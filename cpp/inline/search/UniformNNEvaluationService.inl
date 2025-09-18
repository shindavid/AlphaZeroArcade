#include "search/UniformNNEvaluationService.hpp"

namespace search {

template <search::concepts::Traits Traits>
UniformNNEvaluationService<Traits>::UniformNNEvaluationService() {
  this->set_init_func([](NNEvaluation* eval, const Item& item) {
    eval->uniform_init(item.node()->stable_data().valid_action_mask);
  });
}

}  // namespace search
