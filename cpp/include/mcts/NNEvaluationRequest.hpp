#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/Game.hpp>
#include <cstdint>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/Node.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/FiniteGroups.hpp>
#include <util/Math.hpp>
#include <util/StringUtil.hpp>
#include <util/TinyBitSet.hpp>

#include <span>
#include <string>
#include <vector>

namespace mcts {

// An NNEvaluationRequest is used to make requests to an NNEvaluationService.
//
// The proper usage is for a client to maintain a long-lived NNEvaluationRequest object. Whenever
// the client wishes to request evaluations for one or more positions, it should call
// NNEvaluationRequest::emplace_back() to add the positions to the request. It is important that the
// request is long-lived, because of sensitivities around the reference-counting of NNEvaluation
// objects. The request will hold onto old NNEvaluation objects from previous evaluations, and the
// NNEvaluationService will lazily clear those out when it is safe to do so.
template <core::concepts::Game Game>
class NNEvaluationRequest {
 public:
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using Node = mcts::Node<Game>;
  using InputTensorizor = Game::InputTensorizor;
  using EvalKey = InputTensorizor::EvalKey;
  using NNEvaluation = mcts::NNEvaluation<Game>;

  struct CacheKey {
    CacheKey(const EvalKey& e, group::element_t s);
    CacheKey() = default;
    bool operator==(const CacheKey& c) const { return eval_key == c.eval_key && sym == c.sym; }

    EvalKey eval_key;
    uint64_t hash;
    core::cache_shard_index_t cache_shard;
    group::element_t sym;
  };

  struct CacheKeyHasher {
    size_t operator()(const CacheKey& k) const { return k.hash; }
  };

  class Item {
   public:
    /*
     * The *logical* history represented by this item is given by (using python notation):
     *
     * (history + [state]) if state_is_passed_in else history
     *
     * Passing in a state allows multiple items that share the same history-prefix to share
     * the same history vector, as an optimization.
     *
     * We use sym to transform the history before tensorizing it. If incorporate_sym_into_cache_key
     * is true, then we will incorporate sym into the cache key.
     *
     * The benefit of incorporating sym into the cache key is that when multiple games are played,
     * those games are independent. The downside is that we get less cache hits, hurting game
     * throughput.
     *
     * Based on empirical testing, we find that in self-play, it's better not to incorporate sym
     * into the cache key, to maximize game throughput. The downside is mitigated by the fact that
     * the cache is cleared on each generation, leading to partial-independence. In contrast, for
     * rating games, we incorporate sym into the cache key, to ensure the games are truly
     * independent, in order to get more accurate ratings.
     */
    Item(Node* node, StateHistory& history, const State& state, group::element_t sym,
         bool incorporate_sym_into_cache_key);
    Item(Node* node, StateHistory& history, group::element_t sym,
         bool incorporate_sym_into_cache_key);

    /*
     * Returns f(history.begin(), history.end()),
     *
     * where history is the *logical* history vector associated with this item. See constructor for
     * details.
     */
    template <typename Func>
    auto compute_over_history(Func f) const;

    void set_eval(NNEvaluation* eval) { eval_ = eval; }

    Node* node() const { return node_; }
    NNEvaluation* eval() const { return eval_; }
    const CacheKey& cache_key() const { return cache_key_; }
    group::element_t sym() const { return sym_; }
    const State& cur_state() const { return split_history_ ? state_ : history_->current(); }

   private:
    CacheKey make_cache_key(group::element_t sym, bool incorporate_sym_into_cache_key) const;

    Node* const node_;
    const State state_;
    StateHistory* const history_;
    const bool split_history_;
    const CacheKey cache_key_;
    const group::element_t sym_;

    NNEvaluation* eval_ = nullptr;
  };
  using item_vec_t = std::vector<Item>;

  // In order to alleviate mutex contention, we shard various data structures. We keep S shards,
  // and allocate a given item to shard s iff hash(item) % S == s.
  //
  // NNEvaluationRequest::sub_requests_[shard_index] contains the items that are in shard s.
  class SubRequest {
   public:
    void mark_all_as_stale();
    void push_back(const Item& item) { items_[active_index_].push_back(item); }
    auto stale_items() { return std::span<Item>(items_[1 - active_index_]); }
    auto fresh_items() { return std::span<Item>(items_[active_index_]); }
    void clear_stale_items() { items_[1 - active_index_].clear(); }
    int num_fresh_items() const { return items_[active_index_].size(); }
    int num_stale_items() const { return items_[1 - active_index_].size(); }
    Item& get_fresh_item(int i) { return items_[active_index_][i]; }

   private:
    // We keep two vectors of items: the active vector and the stale vector. After the active items
    // is processed by the NNEvaluationService, they get downgraded to stale.
    //
    // This allows us to better control when items are destroyed (i.e., when the eval reference
    // count gets decremented). That needs to be done while holding the NNEvaluationService::Shard
    // mutex, and in order to do that, we need to destroy lazily.
    item_vec_t items_[2];
    int8_t active_index_ = 0;  // index of the active items_ vector, the other is stale
  };

  void init(search_thread_profiler_t* thread_profiler, int thread_id);

  std::string thread_id_whitespace() const;
  search_thread_profiler_t* thread_profiler() const { return thread_profiler_; }
  int thread_id() const { return thread_id_; }

  template <typename... Ts>
  void emplace_back(Ts&&... args) {
    Item item(std::forward<Ts>(args)...);
    sub_requests_[item.cache_key().cache_shard].push_back(item);
  }

  SubRequest& sub_request(core::cache_shard_index_t shard_index) {
    return sub_requests_[shard_index];
  }

  const SubRequest& sub_request(core::cache_shard_index_t shard_index) const {
    return sub_requests_[shard_index];
  }

  int num_fresh_items() const;
  void mark_all_as_stale();

 private:
  SubRequest sub_requests_[mcts::kNumCacheShards];
  search_thread_profiler_t* thread_profiler_;
  int thread_id_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluationRequest.inl>
