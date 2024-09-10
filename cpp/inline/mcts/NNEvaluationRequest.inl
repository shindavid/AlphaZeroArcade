#include <mcts/NNEvaluationRequest.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluationRequest<Game>::Item::Item(Node* node, const BaseState& state, state_vec_t* history,
                                      group::element_t eval_sym, bool split_history)
    : node_(node),
      state_(state),
      history_(history),
      split_history_(split_history),
      cache_key_(make_cache_key(eval_sym)) {}

template <core::concepts::Game Game>
template <typename Func>
auto NNEvaluationRequest<Game>::Item::compute_over_history(Func f) const {
  util::debug_assert(!history_->empty());
  util::debug_assert(history_->size() <= Game::Constants::kHistorySize + 1);

  if (split_history_) {
    history_->push_back(state_);  // temporary append
  }

  bool full = history_->size() > Game::Constants::kHistorySize + 1;

  auto begin = history_->begin();
  auto end = history_->end();
  if (full) begin++;  // compensate for overstuffing
  auto output = f(&(*begin), &(*(end - 1)));

  if (split_history_) {
    history_->pop_back();  // undo temporary append
  }

  return output;
}

template <core::concepts::Game Game>
typename NNEvaluationRequest<Game>::cache_key_t NNEvaluationRequest<Game>::Item::make_cache_key(
    group::element_t eval_sym) const {
  return std::make_tuple(compute_over_history(InputTensorizor::eval_key), eval_sym);
}

template <core::concepts::Game Game>
NNEvaluationRequest<Game>::NNEvaluationRequest(item_vec_t& items,
                                               search_thread_profiler_t* thread_profiler,
                                               int thread_id)
    : items_(items), thread_profiler_(thread_profiler), thread_id_(thread_id) {}

template <core::concepts::Game Game>
std::string NNEvaluationRequest<Game>::thread_id_whitespace() const {
  return util::make_whitespace(kThreadWhitespaceLength * thread_id_);
}

}  // namespace mcts
