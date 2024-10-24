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
  using SymmetryMask = Game::Types::SymmetryMask;

  using NNEvaluation_sptr = NNEvaluation::sptr;
  using cache_key_t = EvalKey;

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
     * (history + [state]) if first_constructor_used else history
     *
     * The first constructor allows multiple items that share the same history-prefix to share
     * the same history vector, as an optimization.
     */
    Item(Node* node, StateHistory& history, const State& state, const SymmetryMask& sym_mask);
    Item(Node* node, StateHistory& history, const SymmetryMask& sym_mask);

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
    const SymmetryMask& sym_mask() const { return sym_mask_; }
    const State& cur_state() const { return split_history_ ? state_ : history_->current(); }

   private:
    cache_key_t make_cache_key() const;

    Node* const node_;
    const State state_;
    StateHistory* const history_;
    const bool split_history_;
    const cache_key_t cache_key_;
    const SymmetryMask sym_mask_;

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
