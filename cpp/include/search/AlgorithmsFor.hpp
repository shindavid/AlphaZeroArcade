#pragma once

// External mapping from a Traits type to its Algorithms type.
//
// Primary template is intentionally left without a definition so that
// unmapped Traits produce a clear compile-time error at point of use.

namespace search {

template <class Traits>
struct algorithms_for;  // no definition: require a specialization per framework

template <class Traits>
using AlgorithmsForT = algorithms_for<Traits>::type;

}  // namespace search
