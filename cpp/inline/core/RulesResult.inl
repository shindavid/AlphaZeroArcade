#include "core/RulesResult.hpp"

#include "util/Asserts.hpp"

namespace core {

template <typename Traits>
RulesResult<Traits>::RulesResult(const GameOutcome& outcome) : outcome_(outcome), terminal_(true) {}

template <typename Traits>
RulesResult<Traits>::RulesResult(const MoveSet& valid_moves)
    : valid_moves_(valid_moves), terminal_(false) {}

template <typename Traits>
const typename RulesResult<Traits>::GameOutcome& RulesResult<Traits>::outcome() const {
  DEBUG_ASSERT(terminal_);
  return outcome_;
}

template <typename Traits>
const typename RulesResult<Traits>::MoveSet& RulesResult<Traits>::valid_moves() const {
  DEBUG_ASSERT(!terminal_);
  return valid_moves_;
}

}  // namespace core
