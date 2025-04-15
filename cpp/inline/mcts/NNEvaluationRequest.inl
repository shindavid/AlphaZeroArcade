#include <mcts/NNEvaluationRequest.hpp>

#include <util/CppUtil.hpp>

namespace mcts {

template <core::concepts::Game Game>
NNEvaluationRequest<Game>::CacheKey::CacheKey(const EvalKey& e, group::element_t s)
    : eval_key(e), sym(s) {
  // Mix to ensure that the hash is uniformly distributed.
  uint64_t h = math::splitmix64(util::hash(std::make_tuple(e, s)));

  // After mixing, we use the lower bits of the hash to determine the cache shard.
  // The upper bits are used as the object's hash value.
  //
  // It's important to use the upper bits of the mixed-hash, rather than the entire mixed-hash,
  // as otherwise we could have unbalanced buckets in the LRU cache.
  hash = h >> mcts::kCacheShardingFactor;
  cache_shard = h & ((1 << mcts::kCacheShardingFactor) - 1);
}

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
typename NNEvaluationRequest<Game>::CacheKey NNEvaluationRequest<Game>::Item::make_cache_key(
    group::element_t sym, bool incorporate_sym_into_cache_key) const {
  EvalKey eval_key = compute_over_history(
      [&](auto begin, auto end) { return InputTensorizor::eval_key(begin, end - 1); });
  group::element_t cache_sym = incorporate_sym_into_cache_key ? sym : -1;
  return CacheKey(eval_key, cache_sym);
}

template <core::concepts::Game Game>
void NNEvaluationRequest<Game>::init(search_thread_profiler_t* thread_profiler, int thread_id) {
  thread_profiler_ = thread_profiler;
  thread_id_ = thread_id;
}

template <core::concepts::Game Game>
void NNEvaluationRequest<Game>::SubRequest::mark_all_as_stale() {
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

template <core::concepts::Game Game>
std::string NNEvaluationRequest<Game>::thread_id_whitespace() const {
  return util::make_whitespace(kThreadWhitespaceLength * thread_id_);
}

template <core::concepts::Game Game>
int NNEvaluationRequest<Game>::num_fresh_items() const {
  int n = 0;
  for (int i = 0; i < mcts::kNumCacheShards; ++i) {
    n += sub_request(i).num_fresh_items();
  }
  return n;
}

template <core::concepts::Game Game>
void NNEvaluationRequest<Game>::mark_all_as_stale() {
  for (int i = 0; i < mcts::kNumCacheShards; ++i) {
    sub_request(i).mark_all_as_stale();
  }
}

}  // namespace mcts
