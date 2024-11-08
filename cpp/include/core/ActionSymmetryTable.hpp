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
   *
   * The argument is an array that contains concatenated equivalence classes. Any entry that is
   * less than the previous entry is considered the start of a new equivalence class. If a -1 value
   * is encountered, it is considered the end of the array.
   */
  void load(const action_array_t& action_array) { action_array_ = action_array; }

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
