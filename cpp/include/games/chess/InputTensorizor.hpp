#pragma once

#include "core/MultiStateInputTensorizor.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <cstdint>

namespace a0achess {

/*
 * InputTensorizor is based on AlphaZero's input representation:
 *
 * * Dimensions: [103, 8, 8]
 * - History: 8 time steps (Current + 7 past)
 * - Planes per board: 12 (6 Us, 6 Them)
 * - Aux planes: 7 (Castling, Ep, Rule50, etc.)
 *
 * Differences from AlphaZero:
 *
 * - We exclude the repetition plane (seems useless if search treats twofold repetition as a draw)
 * - We exclude the "no-progress" plane (not sure what it is, Lc0 doesn't have it)
 * - We include a plane filled with ones (following Lc0)
 */
struct InputTensorizor : public core::MultiStateInputTensorizorBase<Game, InputFrame, Symmetries,
                                                                    kNumPastFramesToEncode> {
  using Base =
    core::MultiStateInputTensorizorBase<Game, InputFrame, Symmetries, kNumPastFramesToEncode>;

  using EvalKey = zobrist_hash_t;

  static constexpr int kNumFramesToEncode = kNumPastFramesToEncode + 1;
  static constexpr int kPlanesPerBoard = 12;

  enum AuxPlaneIndex : int {
    kAuxPlaneOurQueenSideCastle = 0,
    kAuxPlaneOurKingSideCastle = 1,
    kAuxPlaneTheirQueenSideCastle = 2,
    kAuxPlaneTheirKingSideCastle = 3,
    kAuxPlaneBlackToMove = 4,
    kAuxPlaneRule50PlyCount = 5,
    kAuxPlaneAllOnes = 6,
    kNumAuxPlanes = 7,
  };

  static constexpr int kAuxPlaneBaseIndex = kPlanesPerBoard * kNumFramesToEncode;
  static constexpr int kDim0 = kAuxPlaneBaseIndex + kNumAuxPlanes;

  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;
  using plane_index_t = int;

  Tensor tensorize(group::element_t sym = group::kIdentity);
  void undo();
  void update(const GameState& state);
  void temp_update(const InputFrame& frame);
  EvalKey eval_key() const;

 private:
  void fill_plane(Tensor& tensor, plane_index_t ix, uint64_t data);
  uint64_t current_hash_ = 0;
};

}  // namespace a0achess

#include "inline/games/chess/InputTensorizor.inl"
