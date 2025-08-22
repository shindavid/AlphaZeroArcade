#pragma once

#include "search/SearchContext.hpp"
#include "search/TraitsTypes.hpp"

#include <string>

namespace mcts {

template <typename Traits>
class Algorithms {
 public:
  using Game = Traits::Game;
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using SearchContext = search::SearchContext<Traits>;
  using ValueArray = Game::Types::ValueArray;

  using TraitsTypes = search::TraitsTypes<Traits>;
  using Visitation = TraitsTypes::Visitation;

  using GameResults = Game::GameResults;
  using IO = Game::IO;
  using Symmetries = Game::Symmetries;
  using SymmetryGroup = Game::SymmetryGroup;

  static void pure_backprop(SearchContext& context, const ValueArray& value);
  static void virtual_backprop(SearchContext& context);
  static void undo_virtual_backprop(SearchContext& context);
  static void standard_backprop(SearchContext& context, bool undo_virtual);
  static void short_circuit_backprop(SearchContext& context);

 private:
  static void validate_search_path(const SearchContext& context);
  static std::string search_path_str(const SearchContext& context);
};

}  // namespace mcts

#include "inline/mcts/Algorithms.inl"
