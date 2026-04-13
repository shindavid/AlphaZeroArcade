#include "search/NNEvaluationRequest.hpp"

#include "util/Exceptions.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

template <::alpha0::concepts::Spec Spec>
NNEvaluationRequest<Spec>::Item::Item(const InputFrame& frame, Node* node,
                                            const LookupTable* lookup_table,
                                            const EvalKey& eval_key, InputEncoder& input_encoder,
                                            const InputFrame& extra_frame, group::element_t sym,
                                            bool incorporate_sym_into_cache_key)
    : frame_(frame),
      node_(node),
      lookup_table_(lookup_table),
      extra_frame_(extra_frame),
      input_encoder_(&input_encoder),
      split_history_(true),
      cache_key_(make_cache_key(eval_key, sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <::alpha0::concepts::Spec Spec>
NNEvaluationRequest<Spec>::Item::Item(const InputFrame& frame, Node* node,
                                            const LookupTable* lookup_table,
                                            const EvalKey& eval_key, InputEncoder& input_encoder,
                                            group::element_t sym,
                                            bool incorporate_sym_into_cache_key)
    : frame_(frame),
      node_(node),
      lookup_table_(lookup_table),
      extra_frame_(),
      input_encoder_(&input_encoder),
      split_history_(false),
      cache_key_(make_cache_key(eval_key, sym, incorporate_sym_into_cache_key)),
      sym_(sym) {}

template <::alpha0::concepts::Spec Spec>
template <typename Func>
auto NNEvaluationRequest<Spec>::Item::compute(Func f) const {
  if (split_history_) {
    input_encoder_->temp_update(extra_frame_);  // temporary append
  }

  auto output = f(input_encoder_);

  if (split_history_) {
    input_encoder_->undo();  // undo temporary append
  }

  return output;
}

template <::alpha0::concepts::Spec Spec>
typename NNEvaluationRequest<Spec>::CacheKey
NNEvaluationRequest<Spec>::Item::make_cache_key(const EvalKey& eval_key, group::element_t sym,
                                                      bool incorporate_sym_into_cache_key) const {
  group::element_t cache_sym = incorporate_sym_into_cache_key ? sym : -1;
  return CacheKey(eval_key, cache_sym);
}

template <::alpha0::concepts::Spec Spec>
void NNEvaluationRequest<Spec>::set_notification_task_info(
  const core::YieldNotificationUnit& unit) {
  notification_unit_ = unit;
}

template <::alpha0::concepts::Spec Spec>
void NNEvaluationRequest<Spec>::mark_all_as_stale() {
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
