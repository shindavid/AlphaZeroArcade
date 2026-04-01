#pragma once

#include "util/CompactBitSet.hpp"

#include <string>

namespace core {

// TODO: specify a concept for the Move type (int-constructible, int operator)
template <typename Move, int N>
class BitSetMoveList {
 public:
  static constexpr bool kSortedByMove = true;

  bool operator==(const BitSetMoveList& other) const = default;
  void set_all() { moves_.set(); }

  void add(const Move& move) { moves_.set(int(move)); }
  void remove(const Move& move) { moves_.reset(int(move)); }
  bool contains(const Move& move) const { return moves_.test(int(move)); }
  void clear() { moves_.reset(); }
  int count() const { return moves_.count(); }
  bool empty() const { return moves_.none(); }

  Move get_random(std::mt19937& prng) const;  // assumes !empty()

  auto begin() const;
  auto end() const;

  size_t serialize(char* buffer) const;
  size_t deserialize(const char* buffer);
  std::string to_string() const { return moves_.to_string_natural(); }

 private:
  util::CompactBitSet<N> moves_;
};

}  // namespace core

#include "inline/core/BitSetMoveList.inl"
