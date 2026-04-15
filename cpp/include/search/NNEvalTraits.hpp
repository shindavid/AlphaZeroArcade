#pragma once

#include "core/concepts/TensorEncodingsConcept.hpp"
#include "search/concepts/GraphTraitsConcept.hpp"

namespace search {

template <search::concepts::GraphTraits GraphTraits_,
          core::concepts::TensorEncodings TensorEncodings_, typename NNEvaluation_>
struct NNEvalTraits {
  using GraphTraits = GraphTraits_;
  using TensorEncodings = TensorEncodings_;
  using NNEvaluation = NNEvaluation_;
};

namespace concepts {

template <typename T>
concept NNEvalTraits = requires {
  typename T::GraphTraits;
  typename T::TensorEncodings;
  typename T::NNEvaluation;
  requires search::concepts::GraphTraits<typename T::GraphTraits>;
  requires core::concepts::TensorEncodings<typename T::TensorEncodings>;
};

}  // namespace concepts

}  // namespace search
