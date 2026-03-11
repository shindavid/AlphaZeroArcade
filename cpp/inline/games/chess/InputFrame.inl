#include "games/chess/InputFrame.hpp"
#include <chess-library/include/chess.hpp>

namespace a0achess {

inline InputFrame::InputFrame(const GameState& state) {
  core::seat_index_t cp = (state.sideToMove() == chess::Color::WHITE) ? kWhite : kBlack;

  this->all_pieces[kWhite] = state.us(chess::Color::WHITE);
  this->all_pieces[kBlack] = state.us(chess::Color::BLACK);

  using PieceType = chess::PieceType;
  this->orthogonal_movers = state.pieces(PieceType::ROOK, PieceType::QUEEN);
  this->diagonal_movers = state.pieces(PieceType::BISHOP, PieceType::QUEEN);
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

inline GameState InputFrame::to_state_unsafe() const {
  GameState state(GameState::ProtectedCtor::CREATE);

  state.pieces_bb_[int(chess::PieceType::PAWN)] = getPawns();
  state.pieces_bb_[int(chess::PieceType::KNIGHT)] = getKnights();
  state.pieces_bb_[int(chess::PieceType::BISHOP)] = getBishops();
  state.pieces_bb_[int(chess::PieceType::ROOK)] = getRooks();
  state.pieces_bb_[int(chess::PieceType::QUEEN)] = getQueens();
  state.pieces_bb_[int(chess::PieceType::KING)] = getKings();

  state.occ_bb_[int(chess::Color::WHITE)] = all_pieces[kWhite];
  state.occ_bb_[int(chess::Color::BLACK)] = all_pieces[kBlack];

  state.stm_ = (cur_player == kWhite) ? chess::Color::WHITE : chess::Color::BLACK;

  if (castling_rights & (1 << a0achess::CastlingRightBit::kWhiteKingSide)) {
    state.cr_.setCastlingRight(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE,
                               chess::File::FILE_H);
  }
  if (castling_rights & (1 << a0achess::CastlingRightBit::kWhiteQueenSide)) {
    state.cr_.setCastlingRight(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE,
                               chess::File::FILE_A);
  }
  if (castling_rights & (1 << a0achess::CastlingRightBit::kBlackKingSide)) {
    state.cr_.setCastlingRight(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE,
                               chess::File::FILE_H);
  }
  if (castling_rights & (1 << a0achess::CastlingRightBit::kBlackQueenSide)) {
    state.cr_.setCastlingRight(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE,
                               chess::File::FILE_A);
  }

  chess::Bitboard ep_bb = get_en_passant();
  if (ep_bb) {
    state.ep_sq_ = get_en_passant().pop() ^ 16;
  } else {
    state.ep_sq_ = chess::Square::NO_SQ;
  }

  fill_board(state);

  return state;
}

inline chess::Bitboard InputFrame::get(chess::PieceType piece_type,
                                       core::seat_index_t player) const {
  // clang-format off
  switch (piece_type.internal()) {
    case chess::PieceType::PAWN:   return getPawns(player);
    case chess::PieceType::KNIGHT: return getKnights(player);
    case chess::PieceType::BISHOP: return getBishops(player);
    case chess::PieceType::ROOK:   return getRooks(player);
    case chess::PieceType::QUEEN:  return getQueens(player);
    case chess::PieceType::KING:   return getKings(player);
    default:                       return chess::Bitboard(0);
  }
  // clang-format on
}

inline chess::Bitboard InputFrame::get(chess::PieceType piece_type) const {
  // clang-format off
  switch (piece_type.internal()) {
    case chess::PieceType::PAWN:   return getPawns();
    case chess::PieceType::KNIGHT: return getKnights();
    case chess::PieceType::BISHOP: return getBishops();
    case chess::PieceType::ROOK:   return getRooks();
    case chess::PieceType::QUEEN:  return getQueens();
    case chess::PieceType::KING:   return getKings();
    default:                       return chess::Bitboard(0);
  }
  // clang-format on
}

inline chess::Bitboard InputFrame::getPawns() const {
  return pawns & kPawnsMask;
}

inline chess::Bitboard InputFrame::getKnights() const {
  chess::Bitboard pieces = all_pieces[kWhite] | all_pieces[kBlack];
  chess::Bitboard king_bb = getKings();
  chess::Bitboard pawn_bb = getPawns();
  return pieces & ~(orthogonal_movers | diagonal_movers | king_bb | pawn_bb);
}

inline chess::Bitboard InputFrame::getBishops() const {
  return diagonal_movers & ~orthogonal_movers;
}

inline chess::Bitboard InputFrame::getRooks() const {
  return orthogonal_movers & ~diagonal_movers;
}

inline chess::Bitboard InputFrame::getQueens() const {
  return orthogonal_movers & diagonal_movers;
}

inline chess::Bitboard InputFrame::getKings() const {
  return getKings(kWhite) | getKings(kBlack);
}

inline chess::Bitboard InputFrame::getPawns(core::seat_index_t player) const {
  return getPawns() & all_pieces[player];
}

inline chess::Bitboard InputFrame::getKnights(core::seat_index_t player) const {
  chess::Bitboard pieces = all_pieces[player];
  chess::Bitboard king_bb = getKings(player);
  chess::Bitboard pawn_bb = getPawns();  // no need to pass player here
  return pieces & ~(orthogonal_movers | diagonal_movers | king_bb | pawn_bb);
}

inline chess::Bitboard InputFrame::getBishops(core::seat_index_t player) const {
  return getBishops() & all_pieces[player];
}

inline chess::Bitboard InputFrame::getRooks(core::seat_index_t player) const {
  return getRooks() & all_pieces[player];
}

inline chess::Bitboard InputFrame::getQueens(core::seat_index_t player) const {
  return getQueens() & all_pieces[player];
}

inline chess::Bitboard InputFrame::getKings(core::seat_index_t player) const {
  return chess::Bitboard::fromSquare(chess::Square(static_cast<int>(kings[player])));
}

inline void InputFrame::fill_board(GameState& state) const {
  fill_board(state, kWhite, chess::PieceType::PAWN);
  fill_board(state, kWhite, chess::PieceType::KNIGHT);
  fill_board(state, kWhite, chess::PieceType::BISHOP);
  fill_board(state, kWhite, chess::PieceType::ROOK);
  fill_board(state, kWhite, chess::PieceType::QUEEN);
  fill_board(state, kWhite, chess::PieceType::KING);
  fill_board(state, kBlack, chess::PieceType::PAWN);
  fill_board(state, kBlack, chess::PieceType::KNIGHT);
  fill_board(state, kBlack, chess::PieceType::BISHOP);
  fill_board(state, kBlack, chess::PieceType::ROOK);
  fill_board(state, kBlack, chess::PieceType::QUEEN);
  fill_board(state, kBlack, chess::PieceType::KING);
}

inline void InputFrame::fill_board(GameState& state, core::seat_index_t player,
                               chess::PieceType piece_type) const {
  chess::Bitboard bb = get(piece_type, player);
  chess::Color color = (player == kWhite) ? chess::Color::WHITE : chess::Color::BLACK;
  chess::Piece piece(piece_type, color);
  while (bb) {
    chess::Square sq = bb.pop();
    state.board_[sq.index()] = piece;
  }
}

}  // namespace a0achess
