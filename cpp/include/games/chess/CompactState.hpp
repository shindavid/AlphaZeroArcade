#pragma once

#include "chess-library/include/chess.hpp"
#include "core/BasicTypes.hpp"
#include "games/chess/Constants.hpp"

#include <cstdint>

namespace a0achess {

struct CompactState {
  chess::Bitboard get(chess::PieceType piece_type, core::seat_index_t player) const;
  chess::Bitboard get_en_passant() { return pawns & ~kPawnsMask; }

  chess::Bitboard all_pieces[kNumPlayers];
  chess::Bitboard orthogonal_movers;  // rooks or queens
  chess::Bitboard diagonal_movers;    // bishops or queens

  // Uses the well-known pawn encoding trick, where rank-1 means that white's rank-4 pawn can be
  // taken en passant, and rank-8 means that black's rank-5 pawn can be taken en passant. These
  // pawns are not present in all_pieces.
  chess::Bitboard pawns;

  a0achess::Square kings[kNumPlayers];
  a0achess::CastlingRights castling_rights;
  core::seat_index_t cur_player;
  uint8_t half_move_clock;
};
static_assert(sizeof(CompactState) == 48);

}  // namespace a0achess

#include "inline/games/chess/CompactState.inl"
