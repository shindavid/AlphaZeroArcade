#pragma once

#include "core/BasicTypes.hpp"
#include "search/Constants.hpp"
#include "search/GeneralContext.hpp"
#include "search/NNEvaluationRequest.hpp"
#include "search/RootInfo.hpp"
#include "search/SearchRequest.hpp"
#include "search/TraitsTypes.hpp"
#include "search/concepts/TraitsConcept.hpp"
#include "util/FiniteGroups.hpp"

#include <string>
#include <vector>

namespace search {

// TODO:
// - replace the *history* members with a single InputTensorizor member.
// - remove the leaf_canonical_history member once we drop support for history-utilizing
//   InputTensorizor's with canonical symmetries.
template <search::concepts::Traits Traits>
struct SearchContextBase {
  using Edge = Traits::Edge;
  using Game = Traits::Game;

  using TraitsTypes = search::TraitsTypes<Traits>;
  using Node = TraitsTypes::Node;
  using EvalRequest = search::NNEvaluationRequest<Traits>;
  using GeneralContext = search::GeneralContext<Traits>;
  using RootInfo = search::RootInfo<Traits>;
  using Visitation = TraitsTypes::Visitation;
  using search_path_t = std::vector<Visitation>;

  using StateHistory = TraitsTypes::StateHistory;
  using StateHistoryArray = TraitsTypes::StateHistoryArray;
  using SymmetryGroup = Game::SymmetryGroup;

  int log_prefix_n() const { return kThreadWhitespaceLength * id; }
  void init(const RootInfo&);

  core::context_id_t id;

  GeneralContext* general_context = nullptr;
  search_path_t search_path;

  EvalRequest eval_request;
  StateHistory raw_history;
  StateHistoryArray history_array;      // used in expand_all_children() only
  StateHistory leaf_canonical_history;  // only initialized when needed for nn eval
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

template <search::concepts::Traits Traits, core::TranspositionRule TranspositionRule>
struct SearchContextImpl : public SearchContextBase<Traits> {
  using Base = SearchContextBase<Traits>;
  using Game = Base::Game;
  using Visitation = Base::Visitation;

  std::string search_path_str() const;  // slow, for debugging
};

template <search::concepts::Traits Traits>
struct SearchContextImpl<Traits, core::kSymmetryTranspositions> : public SearchContextBase<Traits> {
  using Base = SearchContextBase<Traits>;
  using Game = Base::Game;
  using RootInfo = Base::RootInfo;
  using SymmetryGroup = Base::SymmetryGroup;
  using Visitation = Base::Visitation;

  std::string search_path_str() const;  // slow, for debugging
  void init(const RootInfo&);

  group::element_t root_canonical_sym;
  group::element_t leaf_canonical_sym;
};

template <search::concepts::Traits Traits>
using SearchContext =
  SearchContextImpl<Traits, Traits::EvalSpec::MctsConfiguration::kTranspositionRule>;

}  // namespace search

#include "inline/search/SearchContext.inl"
