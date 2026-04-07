#include "search/UniformNNEvaluationService.hpp"

namespace search {

template <search::concepts::SearchSpec SearchSpec>
UniformNNEvaluationService<SearchSpec>::UniformNNEvaluationService() {
  this->set_init_func([](NNEvaluation* eval, const Item& item) {
    eval->uniform_init(item.node()->stable_data().num_valid_moves);
  });
}

}  // namespace search
