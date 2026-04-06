#pragma once

#include "util/CompactBitSet.hpp"

#include <string>

namespace core {

// PhasedBitSetMoveList is like BitSetMoveList but also includes a "phase" component in the Move.
//
// This should be used for games where the set of legal moves can be naturally partitioned into a
// small number of "phases", where each moves belongs to exactly one phase, and where the moves in
// each phase can be represented as a bitset.
//
// The Move type must have a two-element constructor (int index, int phase), and an int phase()
// method. TODO: specify a concept for the Move type to be compatible with this class.
template <typename Move, int N>
class PhasedBitSetMoveList {
 public:
  static constexpr bool kSortedByMove = true;

  bool operator==(const PhasedBitSetMoveList& other) const = default;

  void add(const Move& move);
  void remove(const Move& move);
  bool contains(const Move& move) const;
  void clear();
  int size() const { return indices_.count(); }
  bool empty() const { return indices_.none(); }

  Move get_random(std::mt19937& prng) const;  // assumes !empty()

  auto begin() const;
  auto end() const;

  std::string to_string() const;

 private:
  util::CompactBitSet<N> indices_;
  int phase_ = -1;
};

}  // namespace core

#include "inline/core/PhasedBitSetMoveList.inl"
