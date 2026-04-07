#pragma once

#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "search/GeneralContext.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/SearchRequest.hpp"
#include "search/concepts/SearchSpecConcept.hpp"

#include <string>
#include <vector>

namespace search {

template <search::concepts::SearchSpec SearchSpec>
struct SearchContext {
  int log_prefix_n() const { return kThreadWhitespaceLength * id; }
  std::string search_path_str() const;  // slow, for debugging

  using Edge = SearchSpec::Edge;
  using Game = SearchSpec::Game;

  using State = Game::State;
  using Move = Game::Move;
  using Node = SearchSpec::Node;
  using EvalRequest = search::NNEvaluationRequest<SearchSpec>;
  using GeneralContext = search::GeneralContext<SearchSpec>;
  struct Visitation {
    Node* node;
    Edge* edge;  // emanates from node, possibly nullptr
  };
  using search_path_t = std::vector<Visitation>;
  using InputEncoder = SearchSpec::EvalSpec::TensorEncodings::InputEncoder;

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
  const SearchRequest* search_request = nullptr;
};

}  // namespace search

#include "inline/search/SearchContext.inl"
