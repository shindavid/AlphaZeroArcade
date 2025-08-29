#include "search/NodeBase.hpp"

#include "util/BitSet.hpp"

namespace search {

template <typename Traits>
template <typename... Ts>
NodeBase<Traits>::NodeBase(LookupTable* lookup_table, Ts&&... args)
    : NodeBaseCore(std::forward<Ts>(args)...),
      lookup_table_(lookup_table),
      mutex_id_(lookup_table->get_random_mutex_id()) {}

template <typename Traits>
typename Traits::Edge* NodeBase<Traits>::get_edge(int i) const {
  DEBUG_ASSERT(this->first_edge_index_ != -1);
  return lookup_table_->get_edge(this->first_edge_index_ + i);
}

template <typename Traits>
typename Traits::Node* NodeBase<Traits>::get_child(const Edge* edge) const {
  if (edge->child_index < 0) return nullptr;
  return this->lookup_table_->get_node(edge->child_index);
}

// NOTE: this can be switched to use binary search if we'd like
template <typename Traits>
core::node_pool_index_t NodeBase<Traits>::lookup_child_by_action(core::action_t action) const {
  int i = 0;
  for (core::action_t a : bitset_util::on_indices(this->stable_data_.valid_action_mask)) {
    if (a == action) {
      return this->get_edge(i)->child_index;
    }
    ++i;
  }
  return -1;
}

template <typename Traits>
void NodeBase<Traits>::update_child_expand_count(int n) {
  child_expand_count_ += n;
  DEBUG_ASSERT(child_expand_count_ <= this->stable_data_.num_valid_actions);
  if (child_expand_count_ < this->stable_data_.num_valid_actions) return;

  // all children have been expanded, check for triviality
  if (child_expand_count_ == 0) return;
  core::node_pool_index_t first_child_index = get_edge(0)->child_index;
  for (int i = 1; i < this->stable_data_.num_valid_actions; ++i) {
    if (get_edge(i)->child_index != first_child_index) return;
  }

  trivial_ = true;
}

template <typename Traits>
bool NodeBase<Traits>::all_children_edges_initialized() const {
  if (this->stable_data_.num_valid_actions == 0) return true;
  if (this->first_edge_index_ == -1) return false;
  for (int j = 0; j < this->stable_data_.num_valid_actions; ++j) {
    if (get_edge(j)->state != Edge::kExpanded) return false;
  }
  return true;
}

template <typename Traits>
void NodeBase<Traits>::initialize_edges() {
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
