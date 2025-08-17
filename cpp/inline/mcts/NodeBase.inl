#include "mcts/NodeBase.hpp"

#include "util/BitSet.hpp"

namespace mcts {

template <core::concepts::Game Game, typename Derived>
inline NodeBase<Game, Derived>::StableData::StableData(const StateHistory& history,
                                                       core::seat_index_t as)
    : StateData(history.current()) {
  VT.setZero();  // to be set lazily
  VT_valid = false;
  valid_action_mask = Game::Rules::get_legal_moves(history);
  num_valid_actions = valid_action_mask.count();
  action_mode = Game::Rules::get_action_mode(history.current());
  is_chance_node = Game::Rules::is_chance_mode(action_mode);
  active_seat = as;
  terminal = false;
}

template <core::concepts::Game Game, typename Derived>
inline NodeBase<Game, Derived>::StableData::StableData(const StateHistory& history,
                                                       const ValueTensor& game_outcome)
    : StateData(history.current()) {
  VT = game_outcome;
  VT_valid = true;
  num_valid_actions = 0;
  action_mode = -1;
  active_seat = -1;
  terminal = true;
  is_chance_node = false;
}

template <core::concepts::Game Game, typename Derived>
NodeBase<Game, Derived>::NodeBase(LookupTable* table, const StateHistory& history,
                                  core::seat_index_t active_seat)
    : stable_data_(history, active_seat),
      lookup_table_(table),
      mutex_id_(table->get_random_mutex_id()) {}

template <core::concepts::Game Game, typename Derived>
NodeBase<Game, Derived>::NodeBase(LookupTable* table, const StateHistory& history,
                                  const ValueTensor& game_outcome)
    : stable_data_(history, game_outcome),
      lookup_table_(table),
      mutex_id_(table->get_random_mutex_id()) {}

template <core::concepts::Game Game, typename Derived>
Derived* NodeBase<Game, Derived>::get_child(const Edge* edge) const {
  if (edge->child_index < 0) return nullptr;
  return this->lookup_table_->get_node(edge->child_index);
}

// NOTE: this can be switched to use binary search if we'd like
template <core::concepts::Game Game, typename Derived>
node_pool_index_t NodeBase<Game, Derived>::lookup_child_by_action(core::action_t action) const {
  int i = 0;
  for (core::action_t a : bitset_util::on_indices(this->stable_data_.valid_action_mask)) {
    if (a == action) {
      return this->get_edge(i)->child_index;
    }
    ++i;
  }
  return -1;
}

template <core::concepts::Game Game, typename Derived>
void NodeBase<Game, Derived>::update_child_expand_count(int n) {
  child_expand_count_ += n;
  DEBUG_ASSERT(child_expand_count_ <= stable_data_.num_valid_actions);
  if (child_expand_count_ < stable_data_.num_valid_actions) return;

  // all children have been expanded, check for triviality
  if (child_expand_count_ == 0) return;
  node_pool_index_t first_child_index = get_edge(0)->child_index;
  for (int i = 1; i < stable_data_.num_valid_actions; ++i) {
    if (get_edge(i)->child_index != first_child_index) return;
  }

  trivial_ = true;
}

template <core::concepts::Game Game, typename Derived>
bool NodeBase<Game, Derived>::all_children_edges_initialized() const {
  if (stable_data_.num_valid_actions == 0) return true;
  if (first_edge_index_ == -1) return false;
  for (int j = 0; j < stable_data_.num_valid_actions; ++j) {
    if (get_edge(j)->state != kExpanded) return false;
  }
  return true;
}

template <core::concepts::Game Game, typename Derived>
Edge* NodeBase<Game, Derived>::get_edge(int i) const {
  DEBUG_ASSERT(first_edge_index_ != -1);
  return lookup_table_->get_edge(first_edge_index_ + i);
}

template <core::concepts::Game Game, typename Derived>
void NodeBase<Game, Derived>::initialize_edges() {
  int n_edges = stable_data_.num_valid_actions;
  if (n_edges == 0) return;
  first_edge_index_ = lookup_table_->alloc_edges(n_edges);

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(stable_data_.valid_action_mask)) {
    Edge* edge = get_edge(i);
    new (edge) Edge();
    edge->action = action;
    i++;
  }
}

}  // namespace mcts
