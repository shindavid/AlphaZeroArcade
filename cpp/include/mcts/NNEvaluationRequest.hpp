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

  NNEvaluationRequest(Node*, FullState*, std::vector<BaseState>*, search_thread_profiler_t*,
                      int thread_id, group::element_t sym);

  std::string thread_id_whitespace() const {
    return util::make_whitespace(kThreadWhitespaceLength * thread_id_);
  }

  Node* node() const { return node_; }
  FullState* state() const { return state_; }
  std::vector<BaseState>* state_history() const { return state_history_; }
  search_thread_profiler_t* thread_profiler() const { return thread_profiler_; }
  int thread_id() const { return thread_id_; }
  group::element_t sym() const { return sym_; }

 private:
  Node* node_;
  FullState* state_;
  std::vector<BaseState>* state_history_;
  search_thread_profiler_t* thread_profiler_;
  int thread_id_;
  group::element_t sym_;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluationRequest.inl>
