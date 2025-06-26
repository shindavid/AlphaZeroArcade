#include <games/hex/UnionFind.hpp>

#include <algorithm>
#include <utility>

namespace hex {

// initialize each vertex to be its own parent
inline void UnionFind::init() {
  for (vertex_t i = 0; i < kNumVertices; ++i) {
    parent[i] = i;
    rank[i]   = 0;
  }
}

inline vertex_t UnionFind::find(vertex_t x) const {
  if (parent[x] != x) {
    parent[x] = find(parent[x]);  // path compression
  }
  return parent[x];
}

inline void UnionFind::unite(vertex_t a, vertex_t b) {
  a = find(a);
  b = find(b);
  if (a == b) return;
  // union by rank
  if (rank[a] < rank[b]) std::swap(a,b);
  parent[b] = a;
  if (rank[a] == rank[b]) ++rank[a];
}

inline bool UnionFind::connected(vertex_t a, vertex_t b) const { return find(a) == find(b); }

inline void UnionFind::rotate() {
  constexpr vertex_t S = Constants::kNumSquares;
  constexpr vertex_t V = kNumVertices;

  std::array<vertex_t, V> new_parent;
  for (vertex_t a = 0; a < V; ++a) {
    new_parent[rotate_vertex(a)] = rotate_vertex(parent[a]);
  }
  parent = std::move(new_parent);
  std::reverse(rank.begin(), rank.begin() + S);
}

inline vertex_t UnionFind::rotate_vertex(vertex_t v) {
  constexpr vertex_t S = Constants::kNumSquares;
  return (v < S) ? (S - 1 - v) : v;
}

}  // namespace hex
