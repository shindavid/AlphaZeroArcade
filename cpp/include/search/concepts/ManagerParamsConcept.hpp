#pragma once

#include "nnet/NNEvaluationServiceParams.hpp"
#include "search/Constants.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class M>
concept ManagerParams = requires(search::Mode mode) {
  { M(mode) };
  requires std::derived_from<M, nnet::NNEvaluationServiceParams>;
};

}  // namespace concepts
}  // namespace search
