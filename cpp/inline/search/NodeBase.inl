#include "search/NodeBase.hpp"

#include "util/BitSet.hpp"

namespace search {

template <core::concepts::Game Game, typename Derived>
template <typename... Ts>
NodeBase<Game, Derived>::NodeBase(LookupTable* lookup_table, Ts&&... args)
    : NodeBaseCore(std::forward<Ts>(args)...), lookup_table_(lookup_table) {}

template <core::concepts::Game Game, typename Derived>
Edge* NodeBase<Game, Derived>::get_edge(int i) const {
  DEBUG_ASSERT(this->first_edge_index_ != -1);
  return lookup_table_->get_edge(this->first_edge_index_ + i);
}

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
  DEBUG_ASSERT(child_expand_count_ <= this->stable_data_.num_valid_actions);
  if (child_expand_count_ < this->stable_data_.num_valid_actions) return;

  // all children have been expanded, check for triviality
  if (child_expand_count_ == 0) return;
  node_pool_index_t first_child_index = get_edge(0)->child_index;
  for (int i = 1; i < this->stable_data_.num_valid_actions; ++i) {
    if (get_edge(i)->child_index != first_child_index) return;
  }

  trivial_ = true;
}

template <core::concepts::Game Game, typename Derived>
bool NodeBase<Game, Derived>::all_children_edges_initialized() const {
  if (this->stable_data_.num_valid_actions == 0) return true;
  if (this->first_edge_index_ == -1) return false;
  for (int j = 0; j < this->stable_data_.num_valid_actions; ++j) {
    if (get_edge(j)->state != kExpanded) return false;
  }
  return true;
}

template <core::concepts::Game Game, typename Derived>
void NodeBase<Game, Derived>::initialize_edges() {
  int n_edges = this->stable_data_.num_valid_actions;
  if (n_edges == 0) return;
  this->first_edge_index_ = lookup_table_->alloc_edges(n_edges);

  int i = 0;
  for (core::action_t action : bitset_util::on_indices(this->stable_data_.valid_action_mask)) {
    Edge* edge = get_edge(i);
    new (edge) Edge();
    edge->action = action;
    i++;
  }
}

}  // namespace search
