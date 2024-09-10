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
  using BaseState = Game::BaseState;
  using FullState = Game::FullState;
  using Node = mcts::Node<Game>;
  using InputTensorizor = Game::InputTensorizor;
  using EvalKey = InputTensorizor::EvalKey;
  using NNEvaluation = mcts::NNEvaluation<Game>;

  using NNEvaluation_sptr = NNEvaluation::sptr;
  using state_vec_t = Game::Types::state_vec_t;
  using state_vec_iterator_t = state_vec_t::iterator;
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
     * history + ([state] if split_history else [])
     *
     * This split_history option allows multiple items that share the same history-prefix to share
     * the same history vector, as an optimization.
     */
    Item(Node* node, const BaseState& state, state_vec_t* history, group::element_t eval_sym,
         bool split_history);

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

   private:
    cache_key_t make_cache_key(group::element_t eval_sym) const;

    Node* const node_;
    const BaseState state_;
    state_vec_t* const history_;
    const bool split_history_;
    const cache_key_t cache_key_;

    NNEvaluation_sptr eval_;
    eval_state_t eval_state_ = kUnknown;
  };
  using item_vec_t = std::vector<Item>;

  NNEvaluationRequest(item_vec_t& items, search_thread_profiler_t* thread_profiler,
                      int thread_id);

  std::string thread_id_whitespace() const;
  search_thread_profiler_t* thread_profiler() const { return thread_profiler_; }
  int thread_id() const { return thread_id_; }
  item_vec_t& items() const { return items_; }

 private:
  item_vec_t& items_;
  search_thread_profiler_t* const thread_profiler_;
  const int thread_id_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluationRequest.inl>
