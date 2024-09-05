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

  struct aux_data_t {
    std::vector<BaseState>* child_oriented_parent_history;
    BaseState canonical_child_state;
    group::element_t eval_sym;
  };
  using aux_data_vec_t = std::vector<aux_data_t>;

  NNEvaluationRequest(Node*, std::vector<BaseState>*, search_thread_profiler_t*,
                      int thread_id, group::element_t eval_sym);

  void add_aux_data(aux_data_vec_t* vec) { aux_data_vec_ = vec; }

  std::string thread_id_whitespace() const {
    return util::make_whitespace(kThreadWhitespaceLength * thread_id_);
  }

  Node* node() const { return node_; }
  std::vector<BaseState>* state_history() const { return state_history_; }
  search_thread_profiler_t* thread_profiler() const { return thread_profiler_; }
  int thread_id() const { return thread_id_; }
  group::element_t eval_sym() const { return eval_sym_; }
  aux_data_vec_t* aux_data_vec() const { return aux_data_vec_; }

 private:
  Node* node_;
  std::vector<BaseState>* state_history_;
  search_thread_profiler_t* thread_profiler_;
  int thread_id_;
  group::element_t eval_sym_;
  aux_data_vec_t* aux_data_vec_ = nullptr;
};

}  // namespace mcts

#include <inline/mcts/NNEvaluationRequest.inl>
