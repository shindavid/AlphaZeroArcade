#pragma once

#include <concepts>

namespace search {
namespace concepts {

// TODO: add more requirements here. Probably want to extract a base class from NNEvaluation first.
template <class E>
concept Evaluation = requires(const E& const_response) {
  // Return true if the evaluation has not yet been loaded into the response.
  { const_response.pending() } -> std::same_as<bool>;
};

}  // namespace concepts
}  // namespace search
