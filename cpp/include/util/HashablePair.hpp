#pragma once

#include <utility>

#include <util/CppUtil.hpp>

namespace util {

template <typename FirstT, typename SecondT>
struct HashablePair {
  FirstT first;
  SecondT second;

  HashablePair() = default;
  HashablePair(const FirstT& f, const SecondT& s) : first(f), second(s) {}

  bool operator==(const HashablePair& other) const {
    return first == other.first && second == other.second;
  }
};

}  // namespace util

template <typename FirstT, typename SecondT>
struct std::hash<util::HashablePair<FirstT, SecondT>> {
  std::size_t operator()(const util::HashablePair<FirstT, SecondT>& pair) const {
    return util::tuple_hash(std::make_tuple(pair.first, pair.second));
  }
};
