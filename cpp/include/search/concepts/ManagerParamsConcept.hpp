#pragma once

#include "search/Constants.hpp"
#include "search/NNEvaluationServiceParams.hpp"

#include <concepts>

namespace search {
namespace concepts {

template <class M>
concept ManagerParams = requires(search::Mode mode) {
  { M(mode) };
  requires std::derived_from<M, search::NNEvaluationServiceParams>;
};

}  // namespace concepts
}  // namespace search
