#include "games/chess/GameState.hpp"

#include "games/chess/MoveEncoder.hpp"

namespace a0achess {

inline void GameState::backtrack_to(const GameState& prev_state) {
  int n = prev_state.prev_states_.size();
  this->prev_states_.erase(this->prev_states_.begin() + n, this->prev_states_.end());

  this->occ_bb_ = prev_state.occ_bb_;
  this->board_ = prev_state.board_;
  this->key_ = prev_state.key_;
  this->cr_ = prev_state.cr_;
  this->plies_ = prev_state.plies_;
  this->stm_ = prev_state.stm_;
  this->ep_sq_ = prev_state.ep_sq_;
  this->hfm_ = prev_state.hfm_;
  this->chess960_ = prev_state.chess960_;
  this->castling_path = prev_state.castling_path;
}

inline core::action_t GameState::action_from_uci(const std::string& uci) const {
  chess::Move move = chess::uci::uciToMove(*this, uci);
  return move_to_nn_idx(*this, move);
}

inline CompactState GameState::to_compact_state() const {
  CompactState compact_state;

  compact_state.all_pieces[kWhite] = us(chess::Color::WHITE);
  compact_state.all_pieces[kBlack] = us(chess::Color::BLACK);

  compact_state.orthogonal_movers =
    pieces(chess::PieceType::ROOK) | pieces(chess::PieceType::QUEEN);
  compact_state.diagonal_movers =
    pieces(chess::PieceType::BISHOP) | pieces(chess::PieceType::QUEEN);

  compact_state.pawns = pieces(chess::PieceType::PAWN);

  // lc0 en passant encoding trick:
  // The ep-capturable pawn is moved to an impossible rank (rank 1 for white, rank 8 for black)
  // and removed from all_pieces, so its presence there signals en passant availability.
  chess::Square ep_sq = enpassantSq();
  if (ep_sq != chess::Square::NO_SQ) {
    // The capturable pawn sits one rank behind the ep target square
    chess::Square pawn_sq(static_cast<int>(ep_sq.index()) ^ 8);
    chess::Bitboard pawn_bb = chess::Bitboard::fromSquare(pawn_sq);
    auto file = pawn_sq.file();

    // Remove pawn from its real position
    compact_state.pawns ^= pawn_bb;

    if (sideToMove() == chess::Color::BLACK) {
      // White pawn on rank 4 -> encode at rank 1
      chess::Square encoded_sq(file, chess::Rank::RANK_1);
      compact_state.pawns |= chess::Bitboard::fromSquare(encoded_sq);
      compact_state.all_pieces[kWhite] ^= pawn_bb;
    } else {
      // Black pawn on rank 5 -> encode at rank 8
      chess::Square encoded_sq(file, chess::Rank::RANK_8);
      compact_state.pawns |= chess::Bitboard::fromSquare(encoded_sq);
      compact_state.all_pieces[kBlack] ^= pawn_bb;
    }
  }

  compact_state.kings[kWhite] = static_cast<a0achess::Square>(kingSq(chess::Color::WHITE).index());
  compact_state.kings[kBlack] = static_cast<a0achess::Square>(kingSq(chess::Color::BLACK).index());

  compact_state.castling_rights = 0;
  if (cr_.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE))
    compact_state.castling_rights |= (1 << static_cast<uint8_t>(a0achess::CastlingRightBit::kWhiteKingSide));
  if (cr_.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE))
    compact_state.castling_rights |= (1 << static_cast<uint8_t>(a0achess::CastlingRightBit::kWhiteQueenSide));
  if (cr_.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE))
    compact_state.castling_rights |= (1 << static_cast<uint8_t>(a0achess::CastlingRightBit::kBlackKingSide));
  if (cr_.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE))
    compact_state.castling_rights |= (1 << static_cast<uint8_t>(a0achess::CastlingRightBit::kBlackQueenSide));

  compact_state.cur_player = (sideToMove() == chess::Color::WHITE) ? kWhite : kBlack;
  compact_state.half_move_clock = halfMoveClock();

  return compact_state;
}

}  // namespace a0achess
