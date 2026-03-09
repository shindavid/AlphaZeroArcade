#pragma once

#include "core/MultiStateInputTensorizor.hpp"
#include "games/chess/CompactState.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/Game.hpp"
#include "util/EigenUtil.hpp"
#include "util/FiniteGroups.hpp"
#include <cstdint>

namespace a0achess {

struct Keys {
  using TransposeKey = uint64_t;
  using EvalKey = Game::State::zobrist_hash_t;
  using InputTensorizor = core::InputTensorizor<Game>;

  static TransposeKey transpose_key(const Game::State& state) { return state.hash(); }

  static EvalKey eval_key(InputTensorizor* input_tensorizor);
};

// We use Unit = Game::State (which derives from Disservin's chess::Board) as the unit of
// tensorization, allowing us to leverage chess::Board's methods like castlingRights(), pieces(),
// etc. to build the input tensor.
//
// If done naively, this would end up copying chess::Board::prev_states_, which is a std::vector of
// the entire history of the game up to that point. This is undesirable. To avoid this, we have
// build() create a copy of state but with an empty history.
struct TensorizationUnitBuilder {
  using Unit = CompactState;

  static Unit build(const Game::State& state);
};

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
struct InputTensorizor : public core::MultiStateInputTensorizorBase<TensorizationUnitBuilder, Game,
                                                                    kNumPastStatesToEncode> {
  using Base =
    core::MultiStateInputTensorizorBase<TensorizationUnitBuilder, Game, kNumPastStatesToEncode>;

  static constexpr int kNumStatesToEncode = kNumPastStatesToEncode + 1;
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

  static constexpr int kAuxPlaneBaseIndex = kPlanesPerBoard * kNumStatesToEncode;
  static constexpr int kDim0 = kAuxPlaneBaseIndex + kNumAuxPlanes;

  using Unit = TensorizationUnitBuilder::Unit;
  using Keys = a0achess::Keys;
  using Tensor = eigen_util::FTensor<Eigen::Sizes<kDim0, kBoardDim, kBoardDim>>;
  using plane_index_t = int;

  Tensor tensorize(group::element_t sym = group::kIdentity);
  uint64_t current_hash() const;
  void undo();
  void update(const State& state);

 private:
  void fill_plane(Tensor& tensor, plane_index_t ix, uint64_t data);
  uint64_t current_hash_ = 0;
};

}  // namespace a0achess

#include "inline/games/chess/InputTensorizor.inl"
