#include "search/NNEvaluationRequest.hpp"

#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

template <search::concepts::Traits Traits>
NNEvaluationRequest<Traits>::Item::Item(Node* node, InputTensorizor& input_tensorizor,
                                        const State& state, group::element_t sym,
                                        bool incorporate_sym_into_cache_key)
    : node_(node),
      state_(state),
      input_tensorizor_(&input_tensorizor),
      split_history_(true),
      cache_key_(make_cache_key(sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <search::concepts::Traits Traits>
NNEvaluationRequest<Traits>::Item::Item(Node* node, InputTensorizor& input_tensorizor,
                                        group::element_t sym, bool incorporate_sym_into_cache_key)
    : node_(node),
      state_(),
      input_tensorizor_(&input_tensorizor),
      split_history_(false),
      cache_key_(make_cache_key(sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <search::concepts::Traits Traits>
template <typename Func>
auto NNEvaluationRequest<Traits>::Item::compute(Func f) const {
  State cur_state = input_tensorizor_->current_state();
  if (split_history_) {
    input_tensorizor_->update(state_);  // temporary append
  }

  auto output = f(input_tensorizor_);

  if (split_history_) {
    input_tensorizor_->undo(cur_state);  // undo temporary append
  }

  return output;
}

template <search::concepts::Traits Traits>
typename NNEvaluationRequest<Traits>::CacheKey NNEvaluationRequest<Traits>::Item::make_cache_key(
  group::element_t sym, bool incorporate_sym_into_cache_key) const {
  EvalKey eval_key = compute([&](auto tensorizor) { return Keys::eval_key(tensorizor); });
  group::element_t cache_sym = incorporate_sym_into_cache_key ? sym : -1;
  return CacheKey(eval_key, cache_sym);
}

template <search::concepts::Traits Traits>
void NNEvaluationRequest<Traits>::set_notification_task_info(
  const core::YieldNotificationUnit& unit) {
  notification_unit_ = unit;
}

template <search::concepts::Traits Traits>
void NNEvaluationRequest<Traits>::mark_all_as_stale() {
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

}  // namespace search
