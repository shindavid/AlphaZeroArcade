#pragma once

#include "core/MultiFrameInputEncoder.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/Symmetries.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"

#include <cstdint>

namespace a0achess {

struct LcZeroInputEncoder : public core::MultiFrameInputEncoderBase<Game, InputFrame, Symmetries,
                                                                    kNumPastFramesToEncode> {
  using Game = a0achess::Game;
  using Base =
    core::MultiFrameInputEncoderBase<Game, InputFrame, Symmetries, kNumPastFramesToEncode>;

  using EvalKey = zobrist_hash_t;

  static constexpr int kNumFramesToEncode = kNumPastFramesToEncode + 1;
  static constexpr int kPlanesPerBoard = 13;  // MUST be 13 to match AlphaZero/Lc0

  enum AuxPlaneIndex : int {
    kAuxPlaneOurQueenSideCastle = 0,
    kAuxPlaneOurKingSideCastle = 1,
    kAuxPlaneTheirQueenSideCastle = 2,
    kAuxPlaneTheirKingSideCastle = 3,
    kAuxPlaneBlackToMove = 4,
    kAuxPlaneRule50PlyCount = 5,
    kAuxPlaneMoveCountZero = 6,  // MUST pad this empty plane
    kAuxPlaneAllOnes = 7,        // Now shifted to index 7
    kNumAuxPlanes = 8,           // Total 8 aux planes
  };

  static constexpr int kAuxPlaneBaseIndex = kPlanesPerBoard * kNumFramesToEncode;
  static constexpr int kDim0 = kAuxPlaneBaseIndex + kNumAuxPlanes;

  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;
  using plane_index_t = int;

  Tensor encode(group::element_t sym = group::kIdentity);
  void undo();
  void update(const GameState& state);
  void temp_update(const InputFrame& frame);
  EvalKey eval_key() const;

 private:
  void fill_plane(Tensor& tensor, plane_index_t ix, uint64_t data);
  uint64_t current_hash_ = 0;

  inline uint64_t orient_bitboard(uint64_t mask, core::seat_index_t us) {
    // If Black is to move at t=0, flip the board vertically so Black pawns move "up"
    if (us == kBlack) {
      return __builtin_bswap64(mask);
    }
    return mask;
  }
};

}  // namespace a0achess

#include "inline/games/chess/LcZeroInputEncoder.inl"
