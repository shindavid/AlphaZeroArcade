#include "mcts/UniformNNEvaluationService.hpp"

namespace mcts {

template <typename Traits>
UniformNNEvaluationService<Traits>::UniformNNEvaluationService() {
  this->set_init_func([](NNEvaluation* eval, const Item& item) {
    eval->uniform_init(item.node()->stable_data().valid_action_mask);
  });
}

}  // namespace mcts
