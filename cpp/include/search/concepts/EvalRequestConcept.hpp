#pragma once

#include <concepts>

namespace search {
namespace concepts {

// TODO: add more requirements here. Probably want to extract a base class from NNEvaluationRequest
// first.
template <class E>
concept EvalRequest = requires(const E& const_request) {
  typename E::Item;
  { const_request.num_fresh_items() } -> std::same_as<int>;
};

}  // namespace concepts
}  // namespace search
