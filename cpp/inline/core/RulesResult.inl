#include "core/RulesResult.hpp"

#include "util/Asserts.hpp"

namespace core {

template <typename Types>
RulesResult<Types>::RulesResult(const GameOutcome& outcome) : outcome_(outcome), terminal_(true) {}

template <typename Types>
RulesResult<Types>::RulesResult(const MoveSet& valid_moves)
    : valid_moves_(valid_moves), terminal_(false) {}

template <typename Types>
const typename RulesResult<Types>::GameOutcome& RulesResult<Types>::outcome() const {
  DEBUG_ASSERT(terminal_);
  return outcome_;
}

template <typename Types>
const typename RulesResult<Types>::MoveSet& RulesResult<Types>::valid_moves() const {
  DEBUG_ASSERT(!terminal_);
  return valid_moves_;
}

}  // namespace core
