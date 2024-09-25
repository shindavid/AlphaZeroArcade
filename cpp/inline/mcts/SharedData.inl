#include <mcts/SharedData.hpp>

namespace mcts {

template <core::concepts::Game Game>
void SharedData<Game>::clear() {
  root_softmax_temperature.reset();
  lookup_table.clear();
  root_info.node_index = -1;

  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    root_info.history_array[sym].initialize(Rules{});
  }

  const BaseState& raw_state = root_info.history_array[group::kIdentity].current();
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);
}

template <core::concepts::Game Game>
void SharedData<Game>::update_state(core::action_t action) {
  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    core::action_t transformed_action = action;
    Game::Symmetries::apply(transformed_action, sym);
    Game::Rules::apply(root_info.history_array[sym], transformed_action);
  }

  const BaseState& raw_state = root_info.history_array[group::kIdentity].current();
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);
}

}  // namespace mcts
