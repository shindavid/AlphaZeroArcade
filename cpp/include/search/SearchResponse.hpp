#pragma once

#include "core/BasicTypes.hpp"

namespace search {

template <typename Traits>
struct SearchResponse {
  using SearchResults = Traits::SearchResults;

  static SearchResponse make_drop() { return SearchResponse(nullptr, core::kDrop); }
  static SearchResponse make_yield(int e = 0) { return SearchResponse(nullptr, core::kYield, e); }

  SearchResponse(const SearchResults* r, core::yield_instruction_t y = core::kContinue, int e = 0)
      : results(r), yield_instruction(y), extra_enqueue_count(e) {}

  const SearchResults* results;
  core::yield_instruction_t yield_instruction;
  int extra_enqueue_count;
};

}  // namespace search
