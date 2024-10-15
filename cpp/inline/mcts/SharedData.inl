#include <mcts/SharedData.hpp>

namespace mcts {

template <core::concepts::Game Game>
SharedData<Game>::SharedData(const ManagerParams& manager_params, int manager_id)
    : root_softmax_temperature(manager_params.starting_root_softmax_temperature,
                               manager_params.ending_root_softmax_temperature,
                               manager_params.root_softmax_temperature_half_life),
      lookup_table(manager_params.num_search_threads > 1),
      manager_id(manager_id) {
  active_search_threads.resize(manager_params.num_search_threads);
}

template <core::concepts::Game Game>
void SharedData<Game>::clear() {
  root_softmax_temperature.reset();
  lookup_table.clear();
  root_info.node_index = -1;

  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    root_info.history_array[sym].initialize(Rules{});
    Game::Symmetries::apply(root_info.history_array[sym].current(), sym);
  }

  const State& raw_state = root_info.history_array[group::kIdentity].current();
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);
}

template <core::concepts::Game Game>
void SharedData<Game>::update_state(core::action_t action) {
  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    core::action_t transformed_action = action;
    Game::Symmetries::apply(transformed_action, sym);
    Game::Rules::apply(root_info.history_array[sym], transformed_action);
  }

  const State& raw_state = root_info.history_array[group::kIdentity].current();
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);
}

template <core::concepts::Game Game>
void SharedData<Game>::init_root_info(bool add_noise) {
  if (root_info.node_index < 0 || add_noise) {
    const StateHistory& canonical_history = root_info.history_array[root_info.canonical_sym];
    ActionOutcome outcome;
    root_info.node_index = lookup_table.alloc_node();
    Node* root = lookup_table.get_node(root_info.node_index);
    new (root) Node(&lookup_table, canonical_history, outcome);
    root->stats().RN++;
  }
}

}  // namespace mcts
