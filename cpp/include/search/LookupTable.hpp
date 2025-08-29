#pragma once

#include "core/BasicTypes.hpp"
#include "core/NodeBaseCore.hpp"
#include "util/AllocPool.hpp"
#include "util/mit/mit.hpp"  // IWYU pragma: keep

namespace search {

template <typename Traits>
class LookupTable {
 public:
  using Node = Traits::Node;
  using Edge = Traits::Edge;
  using Game = Traits::Game;
  using MCTSKey = Game::InputTensorizor::MCTSKey;
  using NodeBaseCore = core::NodeBaseCore<Game>;

  class Defragmenter {
   public:
    Defragmenter(LookupTable* table);
    void scan(core::node_pool_index_t);
    void prepare();
    void remap(core::node_pool_index_t&);
    void defrag();

   private:
    using bitset_t = boost::dynamic_bitset<>;
    using index_vec_t = std::vector<util::pool_index_t>;

    void remap_helper(core::node_pool_index_t, bitset_t&);
    static void init_remapping(index_vec_t&, bitset_t&);

    LookupTable* table_;
    bitset_t node_bitset_;
    bitset_t edge_bitset_;

    index_vec_t node_index_remappings_;
    index_vec_t edge_index_remappings_;
  };

  LookupTable(core::mutex_vec_sptr_t mutex_pool);
  LookupTable(const LookupTable&) = delete;
  LookupTable& operator=(const LookupTable&) = delete;

  void clear();

  void defragment(core::node_pool_index_t& root_index);

  // Inserts a mapping from k to v.
  //
  // If overwrite is true, the mapping is inserted regardless of whether k is already in the map.
  // Else, the mapping is only inserted if k is not already in the map.
  //
  // Returns the value that k maps to after the operation.
  core::node_pool_index_t insert_node(const MCTSKey& k, core::node_pool_index_t v, bool overwrite);

  // Returns the value that k maps to, or -1 if k is not in the map.
  core::node_pool_index_t lookup_node(const MCTSKey&) const;

  core::node_pool_index_t alloc_node() { return node_pool_.alloc(1); }
  core::edge_pool_index_t alloc_edges(int k) { return edge_pool_.alloc(k); }
  const Node* get_node(core::node_pool_index_t index) const { return &node_pool_[index]; }
  Node* get_node(core::node_pool_index_t index) { return &node_pool_[index]; }
  const Edge* get_edge(core::edge_pool_index_t index) const { return &edge_pool_[index]; }
  Edge* get_edge(core::edge_pool_index_t index) { return &edge_pool_[index]; }

  using map_t = std::unordered_map<MCTSKey, core::node_pool_index_t>;
  const map_t* map() const { return &map_; }

  int get_random_mutex_id() const;
  mit::mutex& get_mutex(int mutex_id);

 private:
  friend class Defragmenter;
  map_t map_;
  util::AllocPool<Edge> edge_pool_;
  util::AllocPool<Node> node_pool_;
  core::mutex_vec_sptr_t mutex_pool_;
  const int mutex_pool_size_;
  mutable mit::mutex map_mutex_;
};

}  // namespace search

#include "inline/search/LookupTable.inl"
