#pragma once

#include "core/concepts/GameConcept.hpp"
#include "util/EigenUtil.hpp"

namespace core {

// ActionPrinter is a utility that is useful for print action-columns in eigen_util::print_array().
template <concepts::Game Game>
class ActionPrinter {
 public:
  static constexpr int kMaxBranchingFactor = Game::Constants::kMaxBranchingFactor;

  using Move = Game::Move;
  using MoveList = Game::MoveList;
  using LocalPolicyArray = eigen_util::DArray<kMaxBranchingFactor>;

  ActionPrinter(const MoveList&);

  void update_format_map(eigen_util::PrintArrayFormatMap&) const;
  const LocalPolicyArray& flat_array() const { return array_; }

 private:
  LocalPolicyArray array_;
  Move moves_[kMaxBranchingFactor];
};

}  // namespace core

#include "inline/core/ActionPrinter.inl"
