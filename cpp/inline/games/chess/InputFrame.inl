#include "games/chess/InputFrame.hpp"

namespace a0achess {

inline InputFrame::InputFrame(const GameState& state) {
  core::seat_index_t cp = (state.sideToMove() == chess::Color::WHITE) ? kWhite : kBlack;

  this->all_pieces[kWhite] = state.us(chess::Color::WHITE);
  this->all_pieces[kBlack] = state.us(chess::Color::BLACK);

  using PieceType = chess::PieceType;
  this->orthogonal_movers = state.pieces(PieceType::ROOK) | state.pieces(PieceType::QUEEN);
  this->diagonal_movers = state.pieces(PieceType::BISHOP) | state.pieces(PieceType::QUEEN);
  this->pawns = state.pieces(PieceType::PAWN);

  // lc0 en passant encoding trick:
  // An ep-capturable pawn marker is added to an impossible rank (rank 1 for white, rank 8 for
  // black)
  chess::Square ep_sq = state.enpassantSq();
  if (ep_sq != chess::Square::NO_SQ) {
    chess::File file = ep_sq.file();
    chess::Rank rank = cp == kBlack ? chess::Rank::RANK_1 : chess::Rank::RANK_8;
    chess::Square encoded_sq(file, rank);
    this->pawns |= chess::Bitboard::fromSquare(encoded_sq);
  }

  this->kings[kWhite] = static_cast<a0achess::Square>(state.kingSq(chess::Color::WHITE).index());
  this->kings[kBlack] = static_cast<a0achess::Square>(state.kingSq(chess::Color::BLACK).index());

  auto cr = state.castlingRights();

  using cC = chess::Color;
  using cS = chess::Board::CastlingRights::Side;
  using aCRB = a0achess::CastlingRightBit;

  this->castling_rights = 0;
  if (cr.has(cC::WHITE, cS::KING_SIDE)) this->castling_rights |= (1 << aCRB::kWhiteKingSide);
  if (cr.has(cC::WHITE, cS::QUEEN_SIDE)) this->castling_rights |= (1 << aCRB::kWhiteQueenSide);
  if (cr.has(cC::BLACK, cS::KING_SIDE)) this->castling_rights |= (1 << aCRB::kBlackKingSide);
  if (cr.has(cC::BLACK, cS::QUEEN_SIDE)) this->castling_rights |= (1 << aCRB::kBlackQueenSide);

  this->cur_player = cp;
  this->half_move_clock = state.halfMoveClock();
}

inline chess::Bitboard InputFrame::get(chess::PieceType piece_type,
                                       core::seat_index_t player) const {
  chess::Bitboard pieces = all_pieces[player];

  switch (piece_type.internal()) {
    case chess::PieceType::KING:
      return chess::Bitboard::fromSquare(chess::Square(static_cast<int>(kings[player])));
    case chess::PieceType::PAWN:
      return pawns & kPawnsMask & pieces;
    case chess::PieceType::QUEEN:
      return orthogonal_movers & diagonal_movers & pieces;
    case chess::PieceType::ROOK:
      return orthogonal_movers & ~diagonal_movers & pieces;
    case chess::PieceType::BISHOP:
      return diagonal_movers & ~orthogonal_movers & pieces;
    case chess::PieceType::KNIGHT: {
      chess::Bitboard king_bb =
        chess::Bitboard::fromSquare(chess::Square(static_cast<int>(kings[player])));
      chess::Bitboard actual_pawns = pawns & kPawnsMask & pieces;
      return pieces & ~(actual_pawns | orthogonal_movers | diagonal_movers | king_bb);
    }
    default:
      return chess::Bitboard(0);
  }
}

}  // namespace a0achess
