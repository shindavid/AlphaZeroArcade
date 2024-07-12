#include <mcts/SharedData.hpp>

namespace mcts {

template <core::concepts::Game Game>
void SharedData<Game>::clear() {
  root_softmax_temperature.reset();
  lookup_table.clear();
  root_info.node_index = -1;

  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    FullState& state = root_info.state[sym];
    base_state_vec_t& state_history = root_info.state_history[sym];

    Game::Rules::init_state(state, sym);
    state_history.clear();
    util::stuff_back<Game::Constants::kHistorySize>(state_history, state);
  }

  group::element_t e = group::kIdentity;
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(root_info.state[e]);
}

template <core::concepts::Game Game>
void SharedData<Game>::update_state(core::action_t action) {
  for (group::element_t sym = 0; sym < SymmetryGroup::kOrder; ++sym) {
    FullState& state = root_info.state[sym];
    base_state_vec_t& state_history = root_info.state_history[sym];

    core::action_t transformed_action = action;
    Game::Symmetries::apply(transformed_action, sym);
    Game::Rules::apply(state, transformed_action);
    util::stuff_back<Game::Constants::kHistorySize>(state_history, state);
  }

  const FullState& raw_state = root_info.state[group::kIdentity];
  root_info.canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);
}

}  // namespace mcts
