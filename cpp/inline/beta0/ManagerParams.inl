#include "beta0/ManagerParams.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
inline ManagerParams<EvalSpec>::ManagerParams(search::Mode m) : Base(m) {
  if (m == search::kCompetition) {
    dirichlet_mult = 0;
    dirichlet_alpha_factor = 0;
    forced_playouts = false;
    starting_root_softmax_temperature = 1;
    ending_root_softmax_temperature = 1;
    root_softmax_temperature_half_life = 1;
  } else if (m == search::kTraining) {
  } else {
    throw util::Exception("Unknown search::Mode: {}", m);
  }
}

}  // namespace beta0
