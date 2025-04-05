#include <mcts/NNEvaluationRequest.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluationRequest<Game>::Item::Item(Node* node, StateHistory& history, const State& state,
                                      group::element_t sym, bool incorporate_sym_into_cache_key)
    : node_(node),
      state_(state),
      history_(&history),
      split_history_(true),
      cache_key_(make_cache_key(sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <core::concepts::Game Game>
NNEvaluationRequest<Game>::Item::Item(Node* node, StateHistory& history, group::element_t sym,
                                      bool incorporate_sym_into_cache_key)
    : node_(node),
      state_(),
      history_(&history),
      split_history_(false),
      cache_key_(make_cache_key(sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <core::concepts::Game Game>
template <typename Func>
auto NNEvaluationRequest<Game>::Item::compute_over_history(Func f) const {
  if (split_history_) {
    history_->update(state_);  // temporary append
  }

  auto begin = history_->begin();
  auto end = history_->end();
  auto output = f(begin, end);

  if (split_history_) {
    history_->undo();  // undo temporary append
  }

  return output;
}

template <core::concepts::Game Game>
typename NNEvaluationRequest<Game>::cache_key_t NNEvaluationRequest<Game>::Item::make_cache_key(
    group::element_t sym, bool incorporate_sym_into_cache_key) const {
  EvalKey eval_key = compute_over_history(
      [&](auto begin, auto end) { return InputTensorizor::eval_key(begin, end - 1); });
  group::element_t cache_sym = incorporate_sym_into_cache_key ? sym : -1;
  return std::make_tuple(eval_key, cache_sym);
}

template <core::concepts::Game Game>
void NNEvaluationRequest<Game>::init(search_thread_profiler_t* thread_profiler, int thread_id) {
  items_.clear();
  thread_profiler_ = thread_profiler;
  thread_id_ = thread_id;
  pending_eval_ = false;
}

// template <core::concepts::Game Game>
// void NNEvaluationRequest<Game>::mark_as_pending_eval() {
//   pending_eval_ = true;
// }

// template <core::concepts::Game Game>
// void NNEvaluationRequest<Game>::notify() {
//   std::unique_lock lock(*mutex_);
//   pending_eval_ = false;
//   cv_->notify_all();
// }

// template <core::concepts::Game Game>
// void NNEvaluationRequest<Game>::wait_for_eval() {
//   if (!pending_eval_) return;
//   std::unique_lock lock(*mutex_);
//   cv_->wait(lock, [&] { return !pending_eval_; });
// }

template <core::concepts::Game Game>
std::string NNEvaluationRequest<Game>::thread_id_whitespace() const {
  return util::make_whitespace(kThreadWhitespaceLength * thread_id_);
}

}  // namespace mcts
