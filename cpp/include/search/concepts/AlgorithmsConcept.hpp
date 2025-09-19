#pragma once

#include "search/Constants.hpp"

#include <concepts>
#include <ostream>

namespace search {
namespace concepts {

// TODO: add detailed comments explaining what each method should do
template <class A, class PolicyTensor, class ValueArray, class SearchContext, class GeneralContext,
          class SearchResults, class Node, class Edge>
concept Algorithms =
  requires(const SearchContext& const_context, SearchContext& non_const_context,
           const PolicyTensor& policy, const ValueArray& value,
           GeneralContext& const_general_context, GeneralContext& non_const_general_context,
           SearchResults& search_results, const Node* node, const Node* root, const Edge* edge,
           search::RootInitPurpose purpose, std::ostream& ss, int n_rows_to_display) {
    { A::pure_backprop(non_const_context, value) };
    { A::virtual_backprop(non_const_context) };
    { A::undo_virtual_backprop(non_const_context) };
    { A::standard_backprop(non_const_context, true) };
    { A::standard_backprop(non_const_context, false) };
    { A::short_circuit_backprop(non_const_context) };
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
