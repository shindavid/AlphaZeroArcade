#pragma once

#include "search/NNEvaluationServiceParams.hpp"
#include "search/Constants.hpp"

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
