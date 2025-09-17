#pragma once

#include "search/AlgorithmsBase.hpp"

namespace alpha0 {

// For now, most of the code lives in AlgorithmsBase, because beta0 is currently just a copy of
// alpha0. As we specialize beta0 more, we should move more code from AlgorithmsBase to
// alpha0::Algorithms.
template <search::concepts::Traits Traits>
class Algorithms : public search::AlgorithmsBase<Traits> {};

}  // namespace alpha0
