#pragma once

#include <core/concepts/Game.hpp>
#include <mcts/Constants.hpp>
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
  using state_vec_t = Game::Types::state_vec_t;

  struct Item {
    BaseState state;
    Node* node;
    state_vec_t* history;
    group::element_t eval_sym;

    /*
     * If true, then the history represented by this item is
     *
     * this->history + [this->state]  # python syntax
     *
     * Else, the history is just this->history.
     *
     * We have this hacky mechanism because K distinct items might share the same first (N-1) items
     * in their history. Rather than making K copies of the history, we just have all K copies
     * point to the same (N-1)-length history, and then have a this->state member to represent the
     * potentially last item in the history.
     */
    bool add_state;
  };
  using item_vec_t = std::vector<Item>;

  NNEvaluationRequest(const item_vec_t& items, search_thread_profiler_t* thread_profiler,
                      int thread_id)
      : items_(items), thread_profiler_(thread_profiler), thread_id_(thread_id) {}

  std::string thread_id_whitespace() const {
    return util::make_whitespace(kThreadWhitespaceLength * thread_id_);
  }

  search_thread_profiler_t* thread_profiler() const { return thread_profiler_; }
  int thread_id() const { return thread_id_; }
  const item_vec_t& items() const { return items_; }

 private:
  const item_vec_t& items_;
  search_thread_profiler_t* const thread_profiler_;
  const int thread_id_;
};

}  // namespace mcts
