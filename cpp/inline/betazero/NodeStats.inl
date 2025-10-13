#include "betazero/NodeStats.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStats<EvalSpec>::NodeStats() {
  Q_min.fill(EvalSpec::Game::GameResults::kMaxValue);
  Q_max.fill(EvalSpec::Game::GameResults::kMinValue);
}

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::init_q(const ValueArray& value, bool pure) {
  Base::init_q(value, pure);
  Q_min = value;
  Q_max = value;
}

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::update_q(const ValueArray& value) {
  Base::update_q(value);

  // element-wise min and max update
  Q_min = Q_min.min(value);
  Q_max = Q_max.max(value);
}

}  // namespace beta0
