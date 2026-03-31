#include "core/RulesResult.hpp"

#include "util/Asserts.hpp"

namespace core {

template <typename Types>
RulesResult<Types> RulesResult<Types>::make_terminal(const GameResultTensor& game_result) {
  RulesResult result;
  result.outcome_ = game_result;
  result.terminal_ = true;
  return result;
}

template <typename Types>
RulesResult<Types> RulesResult<Types>::make_nonterminal(const MoveList& valid_moves) {
  RulesResult result;
  result.valid_moves_ = valid_moves;
  result.terminal_ = false;
  return result;
}

template <typename Types>
const typename RulesResult<Types>::GameResultTensor& RulesResult<Types>::outcome() const {
  DEBUG_ASSERT(terminal_);
  return outcome_;
}

template <typename Types>
const typename RulesResult<Types>::MoveList& RulesResult<Types>::valid_moves() const {
  DEBUG_ASSERT(!terminal_);
  return valid_moves_;
}

}  // namespace core
