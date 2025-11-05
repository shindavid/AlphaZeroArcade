#include "gammazero/NodeStats.hpp"

namespace gamma0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStats<EvalSpec>::NodeStats() {
  Qgamma.fill(0);
  Qgamma_min.fill(EvalSpec::Game::GameResults::kMaxValue);
  Qgamma_max.fill(EvalSpec::Game::GameResults::kMinValue);
  W.fill(0);
  W_max.fill(0);
}

}  // namespace gamma0
