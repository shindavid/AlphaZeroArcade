#include "betazero/NodeStats.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
void NodeStats<EvalSpec>::init_q(const ValueArray& value, bool pure) {
  Base::init_q(value, pure);
  Q_min = value;
  Q_max = value;
}

}  // namespace beta0
