#include "search/SearchContext.hpp"

#include "util/Asserts.hpp"

#include <boost/algorithm/string/join.hpp>

#include <format>
#include <string>
#include <vector>

namespace search {

template <search::concepts::Traits Traits>
std::string SearchContext<Traits>::search_path_str() const {
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
  RELEASE_ASSERT(cur_sym == this->leaf_canonical_sym, "cur_sym={} leaf_canonical_sym={}", cur_sym,
                 this->leaf_canonical_sym);
  return std::format("[{}]", boost::algorithm::join(vec, delim));
}

}  // namespace search
