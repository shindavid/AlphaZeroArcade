#include <mcts/SharedData.hpp>

namespace mcts {

template <core::concepts::Game Game>
SharedData<Game>::SharedData(const ManagerParams& manager_params, int mgr_id)
    : root_softmax_temperature(manager_params.starting_root_softmax_temperature,
                               manager_params.ending_root_softmax_temperature,
                               manager_params.root_softmax_temperature_half_life),
      lookup_table(manager_params.num_search_threads > 1),
      manager_id(mgr_id) {
  active_search_threads.resize(manager_params.num_search_threads);
}

template <core::concepts::Game Game>
void SharedData<Game>::clear() {
  root_softmax_temperature.reset();
  lookup_table.clear();
  root_info.node_index = -1;

  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    root_info.history_array[sym].initialize(Rules{});
    State& state = root_info.history_array[sym].current();
    Game::Symmetries::apply(state, sym);
  }

  const State& raw_state = root_info.history_array[group::kIdentity].current();
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);
}

template <core::concepts::Game Game>
void SharedData<Game>::update_state(core::action_t action) {
  core::action_mode_t mode = get_current_action_mode();
  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    core::action_t transformed_action = action;
    Game::Symmetries::apply(transformed_action, sym, mode);
    Game::Rules::apply(root_info.history_array[sym], transformed_action);
  }

  const State& raw_state = root_info.history_array[group::kIdentity].current();
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);
}

template <core::concepts::Game Game>
void SharedData<Game>::init_root_info(bool add_noise) {
  if (root_info.node_index < 0 || add_noise) {
    const StateHistory& canonical_history = root_info.history_array[root_info.canonical_sym];
    root_info.node_index = lookup_table.alloc_node();
    Node* root = lookup_table.get_node(root_info.node_index);
    core::seat_index_t active_seat = Game::Rules::get_current_player(canonical_history.current());
    util::release_assert(active_seat >= 0 && active_seat < Game::Constants::kNumPlayers);
    root_info.active_seat = active_seat;
    new (root) Node(&lookup_table, canonical_history, active_seat);
    root->stats().RN++;
  }
}

template <core::concepts::Game Game>
core::action_mode_t SharedData<Game>::get_current_action_mode() const {
  return Rules::get_action_mode(root_info.history_array[0].current());
}

}  // namespace mcts
