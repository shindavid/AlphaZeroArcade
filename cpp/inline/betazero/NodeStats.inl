#include "betazero/NodeStats.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStats<EvalSpec>::NodeStats() {
  Qbeta.fill(0);
  Qbeta_min.fill(EvalSpec::Game::GameResults::kMaxValue);
  Qbeta_max.fill(EvalSpec::Game::GameResults::kMinValue);
  W.fill(0);
  W_max.fill(0);
}

}  // namespace beta0
