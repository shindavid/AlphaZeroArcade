#include "search/SearchContext.hpp"

#include <boost/algorithm/string/join.hpp>

#include <format>
#include <string>
#include <vector>

namespace search {

template <search::concepts::Traits Traits, core::TranspositionRule TranspositionRule>
std::string SearchContextImpl<Traits, TranspositionRule>::search_path_str() const {
  std::string delim = Game::IO::action_delimiter();
  std::vector<std::string> vec;
  for (const Visitation& visitation : this->search_path) {
    if (!visitation.edge) continue;
    core::action_mode_t mode = visitation.node->action_mode();
    core::action_t action = visitation.edge->action;
    vec.push_back(Game::IO::action_to_str(action, mode));
  }
  return std::format("[{}]", boost::algorithm::join(vec, delim));
}

template <search::concepts::Traits Traits>
std::string SearchContextImpl<Traits, core::kSymmetryTranspositions>::search_path_str() const {
  // TODO: is this buggy? Shouldn't cur_sym be initialized to the *inverse* of root_canonical_sym?
  // It happens to work in Connect4 because sym == inverse(sym) in the symmetry group C2. Testing
  // in Othello will probably reveal a bug.
  group::element_t cur_sym = root_canonical_sym;
  std::string delim = Game::IO::action_delimiter();
  std::vector<std::string> vec;
  for (const Visitation& visitation : this->search_path) {
    if (!visitation.edge) continue;
    core::action_mode_t mode = visitation.node->action_mode();
    core::action_t action = visitation.edge->action;
    Game::Symmetries::apply(action, cur_sym, mode);
    cur_sym = SymmetryGroup::compose(cur_sym, SymmetryGroup::inverse(visitation.edge->sym));
    vec.push_back(Game::IO::action_to_str(action, mode));
  }
  return std::format("[{}]", boost::algorithm::join(vec, delim));
}

}  // namespace search
