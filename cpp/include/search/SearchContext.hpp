#pragma once

#include "core/BasicTypes.hpp"

#include "search/Constants.hpp"
#include "search/SearchRequest.hpp"
#include "search/TraitsTypes.hpp"
#include "search/TypeDefs.hpp"
#include "util/FiniteGroups.hpp"

namespace search {

template <typename Traits>
struct SearchContext {
  int log_prefix_n() const { return kThreadWhitespaceLength * id; }

  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using EvalRequest = Traits::EvalRequest;

  using TraitsTypes = search::TraitsTypes<Traits>;
  using StateHistoryArray = TraitsTypes::StateHistoryArray;
  using search_path_t = TraitsTypes::search_path_t;

  using StateHistory = Game::StateHistory;

  core::context_id_t id;

  search_path_t search_path;

  EvalRequest eval_request;
  StateHistory canonical_history;
  StateHistoryArray root_history_array;
  StateHistory raw_history;
  core::seat_index_t active_seat;
  group::element_t root_canonical_sym;
  group::element_t leaf_canonical_sym;

  bool mid_expansion = false;
  bool mid_visit = false;
  bool mid_search_iteration = false;
  bool mid_node_initialization = false;
  bool in_visit_loop = false;

  // node-initialization yield info
  StateHistory* initialization_history;
  search::node_pool_index_t initialization_index = -1;
  search::node_pool_index_t inserted_node_index = -1;
  bool expanded_new_node = false;

  // visit yield info
  Node* visit_node;
  Edge* visit_edge;
  StateHistory* history;
  group::element_t inv_canonical_sym;
  bool applied_action = false;

  // For kYield responses
  core::slot_context_vec_t pending_notifications;
  int pending_notifications_mutex_id = 0;

  // For convenience
  const SearchRequest* search_request = nullptr;
};

}  // namespace search
