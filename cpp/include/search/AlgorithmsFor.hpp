#pragma once

// External mapping from a SearchSpec type to its Algorithms type.
//
// Primary template is intentionally left without a definition so that
// unmapped SearchSpec produce a clear compile-time error at point of use.

namespace search {

template <class SearchSpec>
struct algorithms_for;  // no definition: require a specialization per framework

template <class SearchSpec>
using AlgorithmsForT = algorithms_for<SearchSpec>::type;

}  // namespace search
