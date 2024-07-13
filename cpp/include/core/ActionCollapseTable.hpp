#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/GameConstants.hpp>
#include <core/EigenTypes.hpp>
#include <util/FiniteGroups.hpp>

#include <array>

namespace core {

/*
 * In some positions in some games, certain moves are symmetrically equivalent. For example, in
 * the game of Othello, all 4 legal moves in the starting position are symmetrically equivalent.
 *
 * We therefore *collapse* the set of legal actions. Collapsing entails mapping each legal action
 * to a *representative* action. This can be represented by a function f: A -> A, where A is the
 * set of legal actions. We apply action-collapsing wherever possible during MCTS to maximize
 * search efficiency.
 *
 * When MCTS returns a policy, we undo the collapsing, in order to produce symmetrically unbiased
 * policies. ActionCollapseTable is used to facilitate this uncollapsing step.
 *
 * Usage:
 *
 * ActionCollapseTable table;
 * table.load(lookup);  // lookup is an array representing the function f()
 * policy = table.uncollapse(policy);
 */
template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
class ActionCollapseTable {
 public:
  using PolicyTensor = EigenTypes<GameConstants>::PolicyTensor;
  using lookup_table_t = std::array<core::action_t, GameConstants::kNumActions>;
  using count_table_t = std::array<int, GameConstants::kNumActions>;

  void load(const lookup_table_t& lookup);
  PolicyTensor uncollapse(const PolicyTensor& policy) const;

 private:
  lookup_table_t lookup_table_;
  count_table_t count_table_;
};

}  // namespace core

#include <inline/core/ActionCollapseTable.inl>
