#include "search/LookupTable.hpp"

namespace search {

template <search::concepts::Traits Traits>
LookupTable<Traits>::Defragmenter::Defragmenter(LookupTable* table)
    : table_(table),
      node_bitset_(table->node_pool_.size()),
      edge_bitset_(table->edge_pool_.size()) {}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::Defragmenter::scan(core::node_pool_index_t n) {
  if (n < 0 || node_bitset_[n]) return;

  node_bitset_[n] = true;
  NodeBase* node = &table_->node_pool_[n];
  if (!node->edges_initialized()) return;

  core::edge_pool_index_t first_edge_index = node->get_first_edge_index();
  int n_edges = node->stable_data().num_valid_actions;

  edge_bitset_.set(first_edge_index, n_edges, true);
  for (int e = 0; e < n_edges; ++e) {
    Edge* edge = table_->get_edge(node->get_first_edge_index() + e);
    scan(edge->child_index);
  }
}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::Defragmenter::prepare() {
  init_remapping(node_index_remappings_, node_bitset_);
  init_remapping(edge_index_remappings_, edge_bitset_);
}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::Defragmenter::remap(core::node_pool_index_t& n) {
  bitset_t processed_nodes(table_->node_pool_.size());
  remap_helper(n, processed_nodes);
  n = node_index_remappings_[n];
  DEBUG_ASSERT(processed_nodes == node_bitset_);
}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::Defragmenter::defrag() {
  table_->node_pool_.defragment(node_bitset_);
  table_->edge_pool_.defragment(edge_bitset_);

  for (auto it = table_->map_.begin(); it != table_->map_.end();) {
    if (!node_bitset_[it->second]) {
      it = table_->map_.erase(it);
    } else {
      it->second = node_index_remappings_[it->second];
      ++it;
    }
  }
}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::Defragmenter::remap_helper(core::node_pool_index_t n,
                                                          bitset_t& processed_nodes) {
  if (processed_nodes[n]) return;

  processed_nodes[n] = true;
  NodeBase* node = &table_->node_pool_[n];
  if (!node->edges_initialized()) return;

  core::edge_pool_index_t first_edge_index = node->get_first_edge_index();
  int n_edges = node->stable_data().num_valid_actions;

  for (int e = 0; e < n_edges; ++e) {
    Edge* edge = table_->get_edge(node->get_first_edge_index() + e);
    if (edge->child_index < 0) continue;
    remap_helper(edge->child_index, processed_nodes);
    edge->child_index = node_index_remappings_[edge->child_index];
  }

  node->set_first_edge_index(edge_index_remappings_[first_edge_index]);
}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::Defragmenter::init_remapping(index_vec_t& remappings,
                                                            bitset_t& bitset) {
  remappings.resize(bitset.size());
  for (int i = 0; i < (int)bitset.size(); ++i) {
    remappings[i] = -1;
  }

  auto i = bitset.find_first();
  int k = 0;
  while (i != bitset_t::npos) {
    remappings[i] = k++;
    i = bitset.find_next(i);
  }
}

template <search::concepts::Traits Traits>
LookupTable<Traits>::LookupTable(core::mutex_vec_sptr_t mutex_pool)
    : mutex_pool_(mutex_pool), mutex_pool_size_(mutex_pool->size()) {}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::clear() {
  map_.clear();
  edge_pool_.clear();
  node_pool_.clear();
}

template <search::concepts::Traits Traits>
void LookupTable<Traits>::defragment(core::node_pool_index_t& root_index) {
  Defragmenter defragmenter(this);
  defragmenter.scan(root_index);
  defragmenter.prepare();
  defragmenter.remap(root_index);
  defragmenter.defrag();
}

template <search::concepts::Traits Traits>
core::node_pool_index_t LookupTable<Traits>::insert_node(const TransposeKey& key,
                                                              core::node_pool_index_t value,
                                                              bool overwrite) {
  mit::lock_guard lock(map_mutex_);
  if (overwrite) {
    map_[key] = value;
    return value;
  } else {
    auto result = map_.emplace(key, value);
    return result.first->second;
  }
}

template <search::concepts::Traits Traits>
core::node_pool_index_t LookupTable<Traits>::lookup_node(const TransposeKey& key) const {
  mit::lock_guard lock(map_mutex_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    return -1;
  }
  return it->second;
}

template <search::concepts::Traits Traits>
typename Traits::Node* LookupTable<Traits>::get_node(
  core::node_pool_index_t index) const {
  if (index < 0) return nullptr;
  return const_cast<Node*>(&node_pool_[index]);
}

template <search::concepts::Traits Traits>
typename Traits::Edge* LookupTable<Traits>::get_edge(
  core::edge_pool_index_t index) const {
  if (index < 0) return nullptr;
  return const_cast<Edge*>(&edge_pool_[index]);
}

template <search::concepts::Traits Traits>
typename Traits::Edge* LookupTable<Traits>::get_edge(const Node* parent, int n) const {
  int offset = parent->get_first_edge_index();
  DEBUG_ASSERT(offset >= 0);
  return const_cast<Edge*>(&edge_pool_[offset + n]);
}

template <search::concepts::Traits Traits>
mit::mutex* LookupTable<Traits>::get_random_mutex() {
  int mutex_id = mutex_pool_size_ == 1 ? 0 : util::Random::uniform_sample(0, mutex_pool_size_);
  return &(*mutex_pool_)[mutex_id];
}

}  // namespace search
