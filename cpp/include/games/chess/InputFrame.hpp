#pragma once

#include "chess-library/include/chess.hpp"
#include "core/BasicTypes.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/GameState.hpp"
#include "util/CppUtil.hpp"

#include <cstdint>
#include <functional>

namespace a0achess {

struct InputFrame {
  InputFrame() = default;
  InputFrame(const GameState&);
  bool operator==(const InputFrame& other) const = default;

  chess::Bitboard get(chess::PieceType piece_type, core::seat_index_t player) const;
  chess::Bitboard get_en_passant() { return pawns & ~kPawnsMask; }

  chess::Bitboard all_pieces[kNumPlayers];
  chess::Bitboard orthogonal_movers;  // rooks or queens
  chess::Bitboard diagonal_movers;    // bishops or queens

  // Uses the well-known pawn encoding trick, where rank-1 means that white's rank-4 pawn can be
  // taken en passant, and rank-8 means that black's rank-5 pawn can be taken en passant. The rank1
  // and rank8 pawns are not included in all_pieces.
  // Note: The en passant info is NOT encoded in the neural network explicitly for now. We rely on
  // the neural network to infer it from the historical boards.
  // We could potentially add this to an explicit plane as a feature.
  chess::Bitboard pawns;

  a0achess::Square kings[kNumPlayers];
  a0achess::CastlingRights castling_rights;
  core::seat_index_t cur_player;
  uint8_t half_move_clock;
};
static_assert(sizeof(InputFrame) == 48);

}  // namespace a0achess

namespace std {

template <>
struct hash<a0achess::InputFrame> {
  size_t operator()(const a0achess::InputFrame& frame) const {
    return util::PODHash<a0achess::InputFrame>{}(frame);
  }

};

}  // namespace std

#include "inline/games/chess/InputFrame.inl"
