#include "betazero/NodeStats.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStats<EvalSpec>::NodeStats() {
  Q_min.fill(EvalSpec::Game::GameResults::kMaxValue);
  Q_max.fill(EvalSpec::Game::GameResults::kMinValue);
  W.fill(0);
}

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::update_q(const ValueArray& q, const ValueArray& q_sq, bool pure) {
  Base::update_q(q, q_sq, pure);

  // element-wise min and max update
  Q_min = Q_min.min(this->Q);
  Q_max = Q_max.max(this->Q);
}

}  // namespace beta0
