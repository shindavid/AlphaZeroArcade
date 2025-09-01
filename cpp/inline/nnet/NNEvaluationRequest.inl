#include "nnet/NNEvaluationRequest.hpp"

#include "util/Exceptions.hpp"

namespace nnet {

template <core::concepts::Game Game, typename Evaluation>
NNEvaluationRequest<Game, Evaluation>::Item::Item(NodeBase* node, StateHistory& history,
                                                  const State& state, group::element_t sym,
                                                  bool incorporate_sym_into_cache_key)
    : node_(node),
      state_(state),
      history_(&history),
      split_history_(true),
      cache_key_(make_cache_key(sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <core::concepts::Game Game, typename Evaluation>
NNEvaluationRequest<Game, Evaluation>::Item::Item(NodeBase* node, StateHistory& history,
                                                  group::element_t sym,
                                                  bool incorporate_sym_into_cache_key)
    : node_(node),
      state_(),
      history_(&history),
      split_history_(false),
      cache_key_(make_cache_key(sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <core::concepts::Game Game, typename Evaluation>
template <typename Func>
auto NNEvaluationRequest<Game, Evaluation>::Item::compute_over_history(Func f) const {
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

template <core::concepts::Game Game, typename Evaluation>
typename NNEvaluationRequest<Game, Evaluation>::CacheKey
NNEvaluationRequest<Game, Evaluation>::Item::make_cache_key(
  group::element_t sym, bool incorporate_sym_into_cache_key) const {
  EvalKey eval_key = compute_over_history(
    [&](auto begin, auto end) { return InputTensorizor::eval_key(begin, end - 1); });
  group::element_t cache_sym = incorporate_sym_into_cache_key ? sym : -1;
  return CacheKey(eval_key, cache_sym);
}

template <core::concepts::Game Game, typename Evaluation>
void NNEvaluationRequest<Game, Evaluation>::set_notification_task_info(
  const core::YieldNotificationUnit& unit) {
  notification_unit_ = unit;
}

template <core::concepts::Game Game, typename Evaluation>
void NNEvaluationRequest<Game, Evaluation>::mark_all_as_stale() {
  if (items_[active_index_].empty()) return;
  if (items_[1 - active_index_].empty()) {
    active_index_ = 1 - active_index_;
    return;
  }

  // We have both fresh and stale items. We could easily handle this case by moving items from
  // one vector to the other. But, we don't expect that this case should get hit with the current
  // MCTS logic. So, I'm going to just throw an error if we get here.
  throw util::Exception(
    "NNEvaluationRequest::mark_all_as_stale() - both fresh and stale items exist. "
    "This should not happen with the current MCTS logic.");
}

}  // namespace nnet
