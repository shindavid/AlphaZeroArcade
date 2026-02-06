#pragma once

#include "core/BasicTypes.hpp"
#include "core/InputTensorizor.hpp"
#include "search/Constants.hpp"
#include "search/GeneralContext.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/SearchRequest.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"

#include <string>
#include <vector>

namespace search {

template <search::concepts::Traits Traits>
struct SearchContext {
  int log_prefix_n() const { return kThreadWhitespaceLength * id; }
  std::string search_path_str() const;  // slow, for debugging

  using Edge = Traits::Edge;
  using Game = Traits::Game;

  using TraitsTypes = search::TraitsTypes<Traits>;
  using Node = TraitsTypes::Node;
  using EvalRequest = search::NNEvaluationRequest<Traits>;
  using GeneralContext = search::GeneralContext<Traits>;
  using Visitation = TraitsTypes::Visitation;
  using search_path_t = std::vector<Visitation>;
  using InputTensorizor = core::InputTensorizor<Game>;

  core::context_id_t id;

  GeneralContext* general_context = nullptr;
  search_path_t search_path;

  EvalRequest eval_request;
  InputTensorizor input_tensorizor;
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
  bool applied_action = false;

  // For kYield responses
  core::slot_context_vec_t pending_notifications;
  int pending_notifications_mutex_id = 0;

  // For convenience
  const SearchRequest* search_request = nullptr;
};

}  // namespace search

#include "inline/search/SearchContext.inl"
