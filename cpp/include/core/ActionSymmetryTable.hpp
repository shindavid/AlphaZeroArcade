#pragma once

#include "core/concepts/EvalSpecConcept.hpp"
#include "util/AllocPool.hpp"

#include <array>
#include <vector>

namespace core {

/*
 * In some positions in some games, certain moves are symmetrically equivalent. For example, in
 * the game of Othello, all 4 legal moves in the starting position are symmetrically equivalent.
 *
 * Such equivalences are naturally discovered via MCGS mechanics during search. At the end of
 * search, the discovered equivalences are loaded into this data structure.
 */
template <concepts::EvalSpec EvalSpec>
class ActionSymmetryTable {
 public:
  using PolicyEncoding = EvalSpec::TensorEncodings::PolicyEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;
  using InputFrame = EvalSpec::InputFrame;
  using Game = EvalSpec::Game;
  using Group = Game::SymmetryGroup;
  using Move = Game::Move;
  static constexpr int kMaxNumActions = Game::Constants::kMaxBranchingFactor;
  using move_array_t = std::array<Move, kMaxNumActions>;

  struct Item {
    auto operator<=>(const Item&) const = default;
    util::pool_index_t equivalence_class;
    Move move;
  };

  /*
   * Initialize the data structure. This method modifies the passed-in vector.
   *
   * The input is a vector of (equivalence-class, move) pairs. Those moves with matching
   * equivalence-class are grouped together. These groupings are then respected by the symmetrize()
   * and collapse() methods.
   *
   * The argument is a vector that contains items that meet the requirement of the IdActionPair.
   * load() will encode the symmetrically equivalent moves into a compact form:
   * Any entry that is less than the previous entry is considered the start of a new equivalence
   * class. If a -1 value is encountered, it is considered the end of the array.
   */
  void load(std::vector<Item>& items);

  /*
   * Accepts a policy tensor and returns a new policy tensor where the probabilities of
   * symmetrically equivalent moves are averaged.
   */
  PolicyTensor symmetrize(const InputFrame& frame, const PolicyTensor& policy) const;

  /*
   * Accepts a policy tensor and returns a new policy tensor where the probabilities of
   * symmetrically equivalent moves are shifted so that all but one are zero. The choice of which
   * move to keep is consistent across calls.
   */
  PolicyTensor collapse(const InputFrame& frame, const PolicyTensor& policy) const;

  boost::json::array to_json() const;

 private:
  // move_array_ is a compact representation of the move equivalence classes.
  //
  // The equivalence classes are concatenated together. Each maximal increasing subsequence
  // comprises an equivalence class.
  //
  // For any given set of move equivalence classes, there is exactly one way to encode them
  // according to this scheme.
  //
  // Example:
  //
  // Equivalence classes: {1, 2, 5, 8}, {4, 7}, {3}, {6}
  //
  // move_array: {6, 4, 7, 3, 1, 2, 5, 8}
  move_array_t move_array_;
  int num_moves_;
};

}  // namespace core

#include "inline/core/ActionSymmetryTable.inl"
