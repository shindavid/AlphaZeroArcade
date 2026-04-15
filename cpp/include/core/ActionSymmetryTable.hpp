#pragma once

#include "alpha0/concepts/SpecConcept.hpp"

#include <array>
#include <unordered_map>
#include <vector>

namespace core {

/*
 * In some positions in some games, certain moves are symmetrically equivalent. For example, in
 * the game of Othello, all 4 legal moves in the starting position are symmetrically equivalent.
 *
 * Such equivalences can be loaded into this data structure, and then used to symmetrize or collapse
 * policy tensors.
 */
template <::alpha0::concepts::Spec Spec>
class ActionSymmetryTable {
 public:
  static constexpr int kMaxNumActions = Spec::Game::Constants::kMaxBranchingFactor;

  using PolicyEncoding = Spec::TensorEncodings::PolicyEncoding;
  using PolicyTensor = PolicyEncoding::Tensor;
  using InputFrame = Spec::InputFrame;
  using Game = Spec::Game;
  using Move = Game::Move;
  using Symmetries = Spec::Symmetries;
  using move_array_t = std::array<Move, kMaxNumActions>;
  using equivalence_class_t = int;
  using equivalence_class_array_t = std::array<equivalence_class_t, kMaxNumActions>;

  struct Item {
    auto operator<=>(const Item&) const = default;
    equivalence_class_t equivalence_class;
    int move_index;
  };

  // Helper class used to build an ActionSymmetryTable. Recommended to reuse the same Builder object
  // across multiple calls to load() to amortize memory allocations.
  class Builder {
   public:
    void add(const Move& move, const InputFrame& frame);

   protected:
    void clear();

    friend class ActionSymmetryTable;
    using equivalence_class_t = int;
    using move_vec_t = std::vector<Move>;
    using frame_map_t = std::unordered_map<InputFrame, equivalence_class_t>;
    using item_vec_t = std::vector<Item>;

    move_vec_t moves_;
    frame_map_t frame_map_;
    item_vec_t items_;
  };

  /*
   * Initialize the data structure. Modifies and clears the builder object.
   */
  void load(Builder& builder);

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
  int num_equivalence_classes() const { return num_equivalence_classes_; }

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
  equivalence_class_array_t equivalence_class_array_;

  int size_;  // length of the two arrays above
  int num_equivalence_classes_;
};

}  // namespace core

#include "inline/core/ActionSymmetryTable.inl"
