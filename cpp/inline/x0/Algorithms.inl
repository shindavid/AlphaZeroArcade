#include "x0/Algorithms.hpp"

#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include "util/LoggingUtil.hpp"

#include <unordered_map>
#include <vector>

namespace x0 {

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::print_visit_info(const SearchContext& context) {
  if (search::kEnableSearchDebug) {
    const Node* node = context.visit_node;
    LOG_INFO("{:>{}}visit {} seat={}", "", context.log_prefix_n(), context.search_path_str(),
             node->stable_data().active_seat);
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::write_to_training_info(
  bool use_for_training, const ActionResponse& response, const SearchResults* mcts_results,
  core::seat_index_t seat, GameWriteLog_sptr game_log, TrainingInfo& training_info) {
  // TODO: if we have chance-events between player-events, we should compute this bool
  // differently.
  bool previous_used_for_training = game_log->was_previous_entry_used_for_policy_training();

  training_info.clear();
  training_info.frame = mcts_results->frame;
  training_info.active_seat = seat;
  training_info.move = response.get_move();
  training_info.use_for_training = use_for_training;

  if (use_for_training || previous_used_for_training) {
    training_info.policy_target = mcts_results->policy_target;
    training_info.policy_target_valid =
      x0::Algorithms<SearchSpec>::validate_and_symmetrize_policy_target(
        mcts_results, training_info.policy_target);
  }
}

template <search::concepts::SearchSpec SearchSpec>
bool Algorithms<SearchSpec>::validate_and_symmetrize_policy_target(
  const SearchResults* mcts_results, PolicyTensor& target) {
  float sum = eigen_util::sum(target);
  if (mcts_results->provably_lost || sum == 0 || mcts_results->trivial) {
    // python training code will ignore these rows for policy training.
    return false;
  } else {
    target = mcts_results->action_symmetry_table.symmetrize(mcts_results->frame, target);
    target = target / eigen_util::sum(target);
    eigen_util::debug_assert_is_valid_prob_distr(target);
    return true;
  }
}

template <search::concepts::SearchSpec SearchSpec>
void Algorithms<SearchSpec>::load_action_symmetries(const GeneralContext& general_context,
                                                    const Node* root, SearchResults& results) {
  const auto& stable_data = root->stable_data();
  const LookupTable& lookup_table = general_context.lookup_table;
  const State& root_state = general_context.root_info.state;

  using Item = ActionSymmetryTable::Item;
  std::vector<Item> items;
  items.reserve(stable_data.num_valid_moves);

  using equivalence_class_t = int;
  using map_t = std::unordered_map<InputFrame, equivalence_class_t>;
  map_t map;

  State state = root_state;  // copy
  for (int e = 0; e < stable_data.num_valid_moves; ++e) {
    Edge* edge = lookup_table.get_edge(root, e);
    Game::Rules::apply(state, edge->move);
    InputFrame frame(state);
    group::element_t sym = Symmetries::get_canonical_symmetry(frame);
    Symmetries::apply(frame, sym);

    auto [it, inserted] = map.try_emplace(frame, map.size());
    items.emplace_back(it->second, edge->move);
    Game::Rules::backtrack_state(state, root_state);
  }

  results.action_symmetry_table.load(items);
  results.trivial = (map.size() <= 1);
}

template <search::concepts::SearchSpec SearchSpec>
typename Algorithms<SearchSpec>::ActionValueTensor Algorithms<SearchSpec>::apply_mask(
  const ActionValueTensor& values, const PolicyTensor& mask, float invalid_value) {
  using Indices = Eigen::array<Eigen::Index, 2>;
  Indices reshape_dims = {mask.dimensions()[0], 1};
  Indices bcast = {1, values.dimensions()[1]};
  auto reshaped_mask = mask.reshape(reshape_dims).broadcast(bcast);
  auto selector = reshaped_mask > reshaped_mask.constant(0.5f);
  ActionValueTensor invalid_tensor = reshaped_mask.constant(invalid_value);
  return selector.select(values, invalid_tensor);
}

}  // namespace x0
