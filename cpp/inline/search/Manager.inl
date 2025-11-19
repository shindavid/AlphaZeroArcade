#include "core/Constants.hpp"
#include "search/Manager.hpp"

namespace search {

//////////////////////////////////////////////////////
// ManagerImpl<Traits, core::kSimpleTranspositions> //
//////////////////////////////////////////////////////

////////////////////////////////////////////////////////
// ManagerImpl<Traits, core::kSymmetryTranspositions> //
////////////////////////////////////////////////////////

template <search::concepts::Traits Traits>
void ManagerImpl<Traits, core::kSymmetryTranspositions>::update(core::action_t action) {
  group::element_t root_sym = this->root_info()->canonical_sym;

  // this->root_info()->input_tensorizor.update(action);

  State& raw_state = this->root_info()->history.extend();
  core::action_mode_t mode = Game::Rules::get_action_mode(raw_state);
  Game::Rules::apply(raw_state, action);

  this->root_info()->canonical_sym = Game::Symmetries::get_canonical_symmetry(raw_state);

  this->general_context().step();
  core::node_pool_index_t root_index = this->root_info()->node_index;
  if (root_index < 0) return;

  core::action_t transformed_action = action;
  Game::Symmetries::apply(transformed_action, root_sym, mode);

  Node* root = this->lookup_table()->get_node(root_index);
  this->root_info()->node_index = this->lookup_child_by_action(root, transformed_action);
}

//////////////////////////////////////////////////
// ManagerImpl<Traits, core::kNoTranspositions> //
//////////////////////////////////////////////////

}  // namespace search
