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
    Move move = visitation.edge->move;
    vec.push_back(IO::move_to_str(move));
  }
  return std::format("[{}]", boost::algorithm::join(vec, delim));
}

}  // namespace search
