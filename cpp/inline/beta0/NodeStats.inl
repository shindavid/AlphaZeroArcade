#include "beta0/NodeStats.hpp"

namespace beta0 {

template <core::concepts::EvalSpec EvalSpec>
NodeStats<EvalSpec>::NodeStats() {
  Q_min.fill(EvalSpec::TensorEncodings::GameResultEncoding::kMaxValue);
  Q_max.fill(EvalSpec::TensorEncodings::GameResultEncoding::kMinValue);
}

}  // namespace beta0
