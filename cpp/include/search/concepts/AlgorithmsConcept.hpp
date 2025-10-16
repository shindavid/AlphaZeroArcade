#pragma once

#include "search/Constants.hpp"

#include <concepts>
#include <functional>
#include <ostream>

namespace search {
namespace concepts {

// TODO: add detailed comments explaining what each method should do
//
// TODO: add some signatures to concepts::Algorithms:
//
// - write_to_training_info()
// - to_record()
// - compactify_record()
// - to_view()
template <class A, class PolicyTensor, class ValueArray, class SearchContext, class GeneralContext,
          class SearchResults, class Node, class Edge>
concept Algorithms = requires(
  const SearchContext& const_context, SearchContext& non_const_context, const PolicyTensor& policy,
  const ValueArray& value, GeneralContext& const_general_context,
  GeneralContext& non_const_general_context, SearchResults& search_results, Node* node,
  const Node* root, Edge* edge, search::RootInitPurpose purpose, std::ostream& ss,
  int n_rows_to_display, std::function<void()>& func) {
    { A::backprop_helper(node, edge, non_const_general_context.lookup_table, func) };
    { A::init_node_stats_from_terminal(node) };
    { A::init_node_stats_from_nn_eval(node, true) };
    { A::update_node_stats_and_edge(node, edge, true) };
    { A::virtually_update_node_stats(node) };
    { A::virtually_update_node_stats_and_edge(node, edge) };
    { A::undo_virtual_update(node, edge) };
    { A::validate_search_path(const_context) };
    { A::should_short_circuit(edge, node) } -> std::same_as<bool>;
    { A::more_search_iterations_needed(const_general_context, root) } -> std::same_as<bool>;
    { A::init_root_info(non_const_general_context, purpose) };
    { A::get_best_child_index(const_context) } -> std::same_as<int>;
    { A::load_evaluations(non_const_context) };
    { A::to_results(const_general_context, search_results) };
    { A::print_visit_info(const_context) };
    { A::print_mcts_results(ss, policy, search_results, n_rows_to_display) };
  };

}  // namespace concepts
}  // namespace search
