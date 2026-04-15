#pragma once

#include "util/MetaProgramming.hpp"

#include <cstdint>
#include <string_view>

namespace core {

enum SearchParadigm : int8_t { kParadigmAlphaZero, kParadigmBetaZero, kUnknownParadigm };

template <SearchParadigm Paradigm>
struct SearchParadigmTraits {};

template <>
struct SearchParadigmTraits<kParadigmAlphaZero> {
  static constexpr const char* kName = "alpha0";
};

template <>
struct SearchParadigmTraits<kParadigmBetaZero> {
  static constexpr const char* kName = "beta0";
};

// Must match string names in python code. See SearchParadigm enum in py/shared/basic_types.py
inline SearchParadigm parse_search_paradigm(const char* s) {
  std::string_view sv(s);
  SearchParadigm result = kUnknownParadigm;
  mp::constexpr_for<0, kUnknownParadigm, 1>([&](auto I) {
    constexpr SearchParadigm P = static_cast<SearchParadigm>(I.value);
    if (sv == SearchParadigmTraits<P>::kName) result = P;
  });
  return result;
}

}  // namespace core
