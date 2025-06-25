#pragma once

#include <games/hex/Constants.hpp>
#include <games/hex/Types.hpp>

#include <array>
#include <cstdint>

namespace hex {

/*
 * Helper struct used to compute connected components in a Hex board.
 *
 * One UnionFind data structure is maintained per player. The pieces placed by that player are
 * organized into forests, where each tree in the forest represents a connected component of
 * pieces.
 */
struct UnionFind {
  static constexpr vertex_t kVirtualVertex1 = Constants::kNumSquares + 0;
  static constexpr vertex_t kVirtualVertex2 = Constants::kNumSquares + 1;
  static constexpr vertex_t kNumVertices = Constants::kNumSquares + 2;  // +2 for virtual nodes

  auto operator<=>(const UnionFind&) const = default;
  void init();
  vertex_t find(vertex_t x) const;
  void unite(vertex_t a, vertex_t b);
  bool connected(vertex_t a, vertex_t b) const;

 private:
  mutable std::array<vertex_t, kNumVertices> parent;
  mutable std::array<uint8_t, kNumVertices> rank;
};

}  // namespace hex

#include <inline/games/hex/UnionFind.inl>
