#include <games/hex/UnionFind.hpp>

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

}  // namespace hex
