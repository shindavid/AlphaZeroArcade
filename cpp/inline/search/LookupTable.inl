#include "search/LookupTable.hpp"

namespace search {

template <core::concepts::Game Game, typename Derived>
LookupTable<Game, Derived>::Defragmenter::Defragmenter(LookupTable* table)
    : table_(table),
      node_bitset_(table->node_pool_.size()),
      edge_bitset_(table->edge_pool_.size()) {}

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::Defragmenter::scan(node_pool_index_t n) {
  if (n < 0 || node_bitset_[n]) return;

  node_bitset_[n] = true;
  NodeBaseCore* node = &table_->node_pool_[n];
  if (!node->edges_initialized()) return;

  edge_pool_index_t first_edge_index = node->get_first_edge_index();
  int n_edges = node->stable_data().num_valid_actions;

  edge_bitset_.set(first_edge_index, n_edges, true);
  for (int e = 0; e < n_edges; ++e) {
    Edge* edge = table_->get_edge(node->get_first_edge_index() + e);
    scan(edge->child_index);
  }
}

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::Defragmenter::prepare() {
  init_remapping(node_index_remappings_, node_bitset_);
  init_remapping(edge_index_remappings_, edge_bitset_);
}

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::Defragmenter::remap(node_pool_index_t& n) {
  bitset_t processed_nodes(table_->node_pool_.size());
  remap_helper(n, processed_nodes);
  n = node_index_remappings_[n];
  DEBUG_ASSERT(processed_nodes == node_bitset_);
}

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::Defragmenter::defrag() {
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

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::Defragmenter::remap_helper(node_pool_index_t n,
                                                            bitset_t& processed_nodes) {
  if (processed_nodes[n]) return;

  processed_nodes[n] = true;
  NodeBaseCore* node = &table_->node_pool_[n];
  if (!node->edges_initialized()) return;

  edge_pool_index_t first_edge_index = node->get_first_edge_index();
  int n_edges = node->stable_data().num_valid_actions;

  for (int e = 0; e < n_edges; ++e) {
    Edge* edge = table_->get_edge(node->get_first_edge_index() + e);
    if (edge->child_index < 0) continue;
    remap_helper(edge->child_index, processed_nodes);
    edge->child_index = node_index_remappings_[edge->child_index];
  }

  node->set_first_edge_index(edge_index_remappings_[first_edge_index]);
}

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::Defragmenter::init_remapping(index_vec_t& remappings,
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

template <core::concepts::Game Game, typename Derived>
LookupTable<Game, Derived>::LookupTable(search::mutex_vec_sptr_t mutex_pool)
    : mutex_pool_(mutex_pool), mutex_pool_size_(mutex_pool->size()) {}

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::clear() {
  map_.clear();
  edge_pool_.clear();
  node_pool_.clear();
}

template <core::concepts::Game Game, typename Derived>
void LookupTable<Game, Derived>::defragment(node_pool_index_t& root_index) {
  Defragmenter defragmenter(this);
  defragmenter.scan(root_index);
  defragmenter.prepare();
  defragmenter.remap(root_index);
  defragmenter.defrag();
}

template <core::concepts::Game Game, typename Derived>
node_pool_index_t LookupTable<Game, Derived>::insert_node(const MCTSKey& key,
                                                          node_pool_index_t value, bool overwrite) {
  mit::lock_guard lock(map_mutex_);
  if (overwrite) {
    map_[key] = value;
    return value;
  } else {
    auto result = map_.emplace(key, value);
    return result.first->second;
  }
}

template <core::concepts::Game Game, typename Derived>
node_pool_index_t LookupTable<Game, Derived>::lookup_node(const MCTSKey& key) const {
  mit::lock_guard lock(map_mutex_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    return -1;
  }
  return it->second;
}

template <core::concepts::Game Game, typename Derived>
int LookupTable<Game, Derived>::get_random_mutex_id() const {
  return mutex_pool_size_ == 1 ? 0 : util::Random::uniform_sample(0, mutex_pool_size_);
}

template <core::concepts::Game Game, typename Derived>
mit::mutex& LookupTable<Game, Derived>::get_mutex(int mutex_id) {
  return (*mutex_pool_)[mutex_id];
}

}  // namespace search
