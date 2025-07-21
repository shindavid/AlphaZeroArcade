#include "games/hex/UnionFind.hpp"

#include <algorithm>
#include <utility>

namespace hex {

// initialize each vertex to be its own parent
inline void UnionFind::init() {
  for (vertex_t i = 0; i < kNumVertices; ++i) {
    parent_[i] = i;
  }
  std::fill(rank_.begin(), rank_.end(), 0);
}

inline vertex_t UnionFind::find(vertex_t x) const {
  if (parent_[x] != x) {
    parent_[x] = find(parent_[x]);  // path compression
  }
  return parent_[x];
}

inline void UnionFind::unite(vertex_t a, vertex_t b) {
  a = find(a);
  b = find(b);
  if (a == b) return;
  // union by rank
  if (rank_[a] < rank_[b]) std::swap(a,b);
  parent_[b] = a;
  if (rank_[a] == rank_[b]) ++rank_[a];
}

inline bool UnionFind::connected(vertex_t a, vertex_t b) const { return find(a) == find(b); }

inline void UnionFind::rotate() {
  constexpr vertex_t S = Constants::kNumSquares;
  constexpr vertex_t V = kNumVertices;

  std::array<vertex_t, V> new_parent;
  for (vertex_t a = 0; a < V; ++a) {
    new_parent[rotate_vertex(a)] = rotate_vertex(parent_[a]);
  }
  parent_ = std::move(new_parent);
  std::reverse(rank_.begin(), rank_.begin() + S);
  std::swap(rank_[kVirtualVertex1], rank_[kVirtualVertex2]);
}

inline vertex_t UnionFind::rotate_vertex(vertex_t v) {
  constexpr vertex_t S = Constants::kNumSquares;
  constexpr vertex_t arr[2] = {2 * S + 1, S - 1};
  return arr[v < S] - v;
}

}  // namespace hex
