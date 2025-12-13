#include "betazero/NodeStats.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStats<EvalSpec>::NodeStats() {
  plus_shocks.fill(0);
  minus_shocks.fill(0);
  Q_min.fill(EvalSpec::Game::GameResults::kMaxValue);
  Q_max.fill(EvalSpec::Game::GameResults::kMinValue);
  W.fill(0);
  W_max.fill(0);
}

}  // namespace beta0
