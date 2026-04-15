#include "alpha0/SearchContext.hpp"

#include <boost/algorithm/string/join.hpp>

#include <format>
#include <string>
#include <vector>

namespace alpha0 {

template <alpha0::concepts::Spec Spec>
std::string SearchContext<Spec>::search_path_str() const {
  std::string delim = Game::IO::action_delimiter();
  std::vector<std::string> vec;
  for (const Visitation& visitation : this->search_path) {
    if (!visitation.edge) continue;
    Move move = visitation.edge->move;
    vec.push_back(move.to_str());
  }
  return std::format("[{}]", boost::algorithm::join(vec, delim));
}

}  // namespace alpha0
