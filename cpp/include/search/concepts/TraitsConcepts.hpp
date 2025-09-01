#pragma once

#include "search/GeneralContext.hpp"
#include "search/SearchContext.hpp"
#include "search/concepts/AlgorithmsConcept.hpp"
#include "search/concepts/InnerTraitsConcept.hpp"

namespace search {
namespace concepts {

template <class T>
concept Traits = requires {
  requires search::concepts::InnerTraits<T>;
  requires search::concepts::Algorithms<typename T::Algorithms, typename T::Game::Types::ValueArray,
                                        search::SearchContext<T>, search::GeneralContext<T>,
                                        typename T::SearchResults, typename T::Node,
                                        typename T::Edge>;
};

}  // namespace concepts
}  // namespace search
