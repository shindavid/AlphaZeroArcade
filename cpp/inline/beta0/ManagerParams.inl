#include "beta0/ManagerParams.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
inline ManagerParams<EvalSpec>::ManagerParams(search::Mode m) : Base(m) {
  if (m == search::kCompetition) {
    enable_exploratory_visits = false;
  } else if (m == search::kTraining) {
    enable_exploratory_visits = true;
  } else {
    throw util::Exception("Unknown search::Mode: {}", m);
  }
}

}  // namespace beta0
