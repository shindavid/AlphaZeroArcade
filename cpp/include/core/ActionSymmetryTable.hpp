#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/GameConstants.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <array>
#include <map>
#include <vector>

namespace core {

/*
 * An IdActionPair is a pair of an integer group_id and an action. The group_id is used to group
 * actions that are symmetrically equivalent. The ActionSymmetryTable data structure is a generic
 * way to encode these equivalence classes. A group id represents a set of actions that are
 * symmetrically equivalent. In the current case, we use the pool_index_t type, which is the type
 * used to index into the pool of nodes in the MCTS search. However, group id could be any integer
 * type, as long as it is convertible to an int.
 */
template <typename T>
concept IdActionPair = requires(T t) {
  { t <=> t } -> std::same_as<std::strong_ordering>;
  { t.group_id } -> std::convertible_to<int>;
  { t.action } -> std::convertible_to<core::action_t>;
};

/*
 * In some positions in some games, certain moves are symmetrically equivalent. For example, in
 * the game of Othello, all 4 legal moves in the starting position are symmetrically equivalent.
 *
 * Such equivalences are naturally discovered via MCGS mechanics during search. At the end of
 * search, the discovered equivalences are loaded into this data structure.
 */
template <concepts::GameConstants GameConstants, group::concepts::FiniteGroup Group>
class ActionSymmetryTable {
 public:
  using action_array_t = std::array<core::action_t, GameConstants::kNumActions>;
  using PolicyTensor = eigen_util::FTensor<Eigen::Sizes<GameConstants::kNumActions>>;

  /*
   * Initialize the data structure.
   * The argument is a vector that contains items that meet the requirement of the IdActionPair.
   * load() will encode the symmetrically equivalent actions into a compact form:
   * Any entry that is less than the previous entry is considered the start of a new equivalence
   * class. If a -1 value is encountered, it is considered the end of the array.
   */
  template <IdActionPair item_t>
  void load(std::vector<item_t>& items);

  /*
   * Accepts a policy tensor and returns a new policy tensor where the probabilities of
   * symmetrically equivalent actions are averaged.
   */
  PolicyTensor symmetrize(const PolicyTensor& policy) const;

  /*
   * Accepts a policy tensor and returns a new policy tensor where the probabilities of
   * symmetrically equivalent actions are shifted so that all but one are zero.
   */
  PolicyTensor collapse(const PolicyTensor& policy) const;

  boost::json::array to_json() const;
 private:
  action_array_t action_array_;
};

}  // namespace core

#include <inline/core/ActionSymmetryTable.inl>
