#include "betazero/NodeStats.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStats<EvalSpec>::NodeStats() {
  Q_min.fill(EvalSpec::Game::GameResults::kMaxValue);
  Q_max.fill(EvalSpec::Game::GameResults::kMinValue);
}

}  // namespace beta0
