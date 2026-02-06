#include "x0/Algorithms.hpp"

#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include "util/LoggingUtil.hpp"

#include <unordered_map>
#include <vector>

namespace x0 {

template <search::concepts::Traits Traits>
void Algorithms<Traits>::print_visit_info(const SearchContext& context) {
  if (search::kEnableSearchDebug) {
    const Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), context.search_path_str(),
             node->stable_data().active_seat);
  }
}

template <search::concepts::Traits Traits>
bool Algorithms<Traits>::validate_and_symmetrize_policy_target(const SearchResults* mcts_results,
                                                               PolicyTensor& target) {
  float sum = eigen_util::sum(target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    // python training code will ignore these rows for policy training.
    return false;
  } else {
    target = mcts_results->action_symmetry_table.symmetrize(target);
    target = target / eigen_util::sum(target);
    eigen_util::debug_assert_is_valid_prob_distr(target);
    return true;
  }
}

template <search::concepts::Traits Traits>
void Algorithms<Traits>::load_action_symmetries(const GeneralContext& general_context,
                                                const Node* root, core::action_t* actions,
                                                SearchResults& results) {
  const auto& stable_data = root->stable_data();
  const LookupTable& lookup_table = general_context.lookup_table;
  const State& root_state = general_context.root_info.input_tensorizor.current_state();

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_actions);

  using equivalence_class_t = int;
  using map_t = std::unordered_map<State, equivalence_class_t>;
  map_t map;

  for (int e = 0; e < stable_data.num_valid_actions; ++e) {
    State state = root_state;
    Edge* edge = lookup_table.get_edge(root, e);
    Game::Rules::apply(state, edge->action);
    group::element_t sym = Game::Symmetries::get_canonical_symmetry(state);
    Game::Symmetries::apply(state, sym);

    auto [it, inserted] = map.try_emplace(state, map.size());
    items.emplace_back(it->second, actions[e]);
  }

  results.action_symmetry_table.load(items);
  results.trivial = (map.size() <= 1);
}

}  // namespace x0
