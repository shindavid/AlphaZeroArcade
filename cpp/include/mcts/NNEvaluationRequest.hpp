#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
#include <mcts/NNEvaluation.hpp>
#include <mcts/Node.hpp>
#include <mcts/TypeDefs.hpp>
#include <util/FiniteGroups.hpp>
#include <util/StringUtil.hpp>

#include <string>
#include <vector>

namespace mcts {

template <core::concepts::Game Game>
class NNEvaluationRequest {
 public:
  using State = Game::State;
  using StateHistory = Game::StateHistory;
  using Node = mcts::Node<Game>;
  using InputTensorizor = Game::InputTensorizor;
  using EvalKey = InputTensorizor::EvalKey;
  using NNEvaluation = mcts::NNEvaluation<Game>;

  using NNEvaluation_sptr = NNEvaluation::sptr;

  // If group::element_t is -1, that means to pick a random symmetry at the time of evaluation.
  // Otherwise, it is the index of the symmetry to use.
  using cache_key_t = std::tuple<EvalKey, group::element_t>;

  enum eval_state_t : int8_t {
    kUnknown = 0,
    kClaimedByMe = 1,
    kClaimedByOther = 2,
    kCompleted = 3
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

    void set_eval(NNEvaluation_sptr eval) { eval_ = eval; }
    void set_eval_state(eval_state_t eval_state) { eval_state_ = eval_state; }

    Node* node() const { return node_; }
    NNEvaluation* eval() const { return eval_.get(); }
    eval_state_t eval_state() const { return eval_state_; }
    const cache_key_t& cache_key() const { return cache_key_; }
    group::element_t sym() const { return sym_; }
    const State& cur_state() const { return split_history_ ? state_ : history_->current(); }

   private:
    cache_key_t make_cache_key(group::element_t sym, bool incorporate_sym_into_cache_key) const;

    Node* const node_;
    const State state_;
    StateHistory* const history_;
    const bool split_history_;
    const cache_key_t cache_key_;
    const group::element_t sym_;

    NNEvaluation_sptr eval_;
    eval_state_t eval_state_ = kUnknown;
  };
  using item_vec_t = std::vector<Item>;

  void init(search_thread_profiler_t* thread_profiler, int thread_id);

  // void mark_as_pending_eval();
  // void notify();
  // void wait_for_eval();

  std::string thread_id_whitespace() const;
  search_thread_profiler_t* thread_profiler() const { return thread_profiler_; }
  int thread_id() const { return thread_id_; }
  item_vec_t& items() { return items_; }

 private:
  item_vec_t items_;
  search_thread_profiler_t* thread_profiler_;
  int thread_id_;
  bool pending_eval_ = false;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluationRequest.inl>
