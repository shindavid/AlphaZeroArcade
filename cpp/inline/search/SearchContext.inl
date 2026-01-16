#include "search/SearchContext.hpp"

#include <boost/algorithm/string/join.hpp>

#include <format>
#include <string>
#include <vector>

namespace search {

template <search::concepts::Traits Traits>
std::string SearchContext<Traits>::search_path_str() const {
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

}  // namespace search
