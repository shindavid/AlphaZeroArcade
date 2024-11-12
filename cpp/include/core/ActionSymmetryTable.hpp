#pragma once

#include <core/BasicTypes.hpp>
#include <core/concepts/GameConstants.hpp>
#include <util/AllocPool.hpp>
#include <util/EigenUtil.hpp>
#include <util/FiniteGroups.hpp>

#include <array>
#include <map>
#include <vector>

namespace core {

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

  struct item_t {
    auto operator<=>(const item_t&) const = default;
    util::pool_index_t equivalence_class;
    core::action_t action;
  };

  /*
   * Initialize the data structure. This method modifies the passed-in vector.
   *
   * The input is a vector of (equivalence-class, action) pairs. Those actions with matching
   * equivalence-class are grouped together. These groupings are then respected by the symmetrize()
   * and collapse() methods.
   *
   * The argument is a vector that contains items that meet the requirement of the IdActionPair.
   * load() will encode the symmetrically equivalent actions into a compact form:
   * Any entry that is less than the previous entry is considered the start of a new equivalence
   * class. If a -1 value is encountered, it is considered the end of the array.
   */
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

  // action_array_ is a compact representation of the acton equivalence classes.
  //
  // The equivalence classes are concatenated together. Each maximal increasing subsequence
  // comprises an equivalence class. The array is terminated by a -1.
  //
  // For any given set of action equivalence classes, there is exactly one way to encode them
  // according to this scheme.
  //
  // Example:
  //
  // Equivalence classes: {1, 2, 5, 8}, {4, 7}, {3}, {6}
  //
  // action_array: {6, 4, 7, 3, 1, 2, 5, 8}
  action_array_t action_array_;
};

}  // namespace core

#include <inline/core/ActionSymmetryTable.inl>
