#include <mcts/NNEvaluationRequest.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluationRequest<Game>::NNEvaluationRequest(Node* node,
                                               std::vector<BaseState>* state_history,
                                               search_thread_profiler_t* thread_profiler,
                                               int thread_id, group::element_t eval_sym) {
  node_ = node;
  state_history_ = state_history;
  thread_profiler_ = thread_profiler;
  thread_id_ = thread_id;
  eval_sym_ = eval_sym;
}

}  // namespace mcts
