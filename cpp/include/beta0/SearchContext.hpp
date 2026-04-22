#pragma once

#include "beta0/Edge.hpp"
#include "beta0/GraphTraits.hpp"
#include "beta0/Node.hpp"
#include "beta0/concepts/SpecConcept.hpp"
#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "search/NNEvalTraits.hpp"
#include "search/NNEvaluation.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/SearchRequest.hpp"

#include <string>
#include <vector>

namespace beta0 {

template <beta0::concepts::Spec Spec>
struct SearchContext {
  int log_prefix_n() const { return search::kThreadWhitespaceLength * id; }
  std::string search_path_str() const;  // slow, for debugging

  using Edge = beta0::Edge<Spec>;
  using Game = Spec::Game;

  using State = Game::State;
  using Move = Game::Move;
  using Node = beta0::Node<Spec>;
  using InputFrame = Spec::InputFrame;
  using GraphTraits = beta0::GraphTraits<Spec>;
  using TensorEncodings = Spec::TensorEncodings;
  using NetworkHeads = Spec::NetworkHeads;
  using NNEvaluation = search::NNEvaluation<Game, InputFrame, NetworkHeads>;
  using NNEvalTraits = search::NNEvalTraits<GraphTraits, TensorEncodings, NNEvaluation>;
  using EvalRequest = search::NNEvaluationRequest<NNEvalTraits>;
  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
  using search_path_t = std::vector<Visitation>;
  using InputEncoder = Spec::TensorEncodings::InputEncoder;

  core::context_id_t id;

  search_path_t search_path;

  EvalRequest eval_request;
  InputEncoder input_encoder;
  State current_state;

  // If state_step == root_info.state_step, then we are able to reset current_state
  // to root_info.state via Game::Rules::backtrack_state(). If not, then we are
  // forced to do operator= instead (which is more expensive for chess).
  int state_step;

  core::seat_index_t active_seat;

  bool mid_expansion = false;
  bool mid_visit = false;
  bool mid_search_iteration = false;
  bool mid_node_initialization = false;
  bool in_visit_loop = false;

  // node-initialization yield info
  core::node_pool_index_t initialization_index = -1;
  core::node_pool_index_t inserted_node_index = -1;
  bool expanded_new_node = false;

  // visit yield info
  Node* visit_node;
  Edge* visit_edge;
  bool applied_move = false;

  // For kYield responses
  core::slot_context_vec_t pending_notifications;
  int pending_notifications_mutex_id = 0;

  // For convenience
  const search::SearchRequest* search_request = nullptr;
};

}  // namespace beta0

#include "inline/beta0/SearchContext.inl"
