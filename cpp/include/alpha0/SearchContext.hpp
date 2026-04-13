#pragma once

#include "alpha0/GeneralContext.hpp"
#include "alpha0/Spec.hpp"
#include "core/BasicTypes.hpp"
#include "alpha0/concepts/SpecConcept.hpp"
#include "search/Constants.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/SearchRequest.hpp"

#include <string>
#include <vector>

namespace alpha0 {

template <alpha0::concepts::Spec EvalSpec>
struct SearchContext {
  using Spec = alpha0::Spec<typename EvalSpec::Game, EvalSpec>;

  int log_prefix_n() const { return search::kThreadWhitespaceLength * id; }
  std::string search_path_str() const;  // slow, for debugging

  using Edge = Spec::Edge;
  using Game = EvalSpec::Game;

  using State = Game::State;
  using Move = Game::Move;
  using Node = Spec::Node;
  using EvalRequest = search::NNEvaluationRequest<Spec>;
  using GeneralContext = alpha0::GeneralContext<EvalSpec>;
  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
  using search_path_t = std::vector<Visitation>;
  using InputEncoder = EvalSpec::TensorEncodings::InputEncoder;

  core::context_id_t id;

  GeneralContext* general_context = nullptr;
  search_path_t search_path;

  EvalRequest eval_request;
  InputEncoder input_encoder;
  State current_state;

  // If state_step == general_context.root_info.state_step, then we are able to reset current_state
  // to general_context.root_info.state via Game::Rules::backtrack_state(). If not, then we are
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

}  // namespace alpha0

#include "inline/alpha0/SearchContext.inl"
