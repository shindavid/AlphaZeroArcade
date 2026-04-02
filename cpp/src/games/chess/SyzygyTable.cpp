#include "games/chess/SyzygyTable.hpp"
#include "games/chess/Game.hpp"

namespace a0achess {

namespace {
chess::PieceType fathom_promo_to_piece_type(unsigned promotes) {
  switch (promotes) {
    case TB_PROMOTES_QUEEN:
      return chess::PieceType::QUEEN;
    case TB_PROMOTES_ROOK:
      return chess::PieceType::ROOK;
    case TB_PROMOTES_BISHOP:
      return chess::PieceType::BISHOP;
    case TB_PROMOTES_KNIGHT:
      return chess::PieceType::KNIGHT;
    default:
      return chess::PieceType::NONE;
  }
}
} // namespace

// Convert a Fathom root result to a chess::Move, then to action_t.
a0achess::Move SyzygyTable::fathom_result_to_move(const a0achess::GameState& board, unsigned result) {
  unsigned from_sq = TB_GET_FROM(result);
  unsigned to_sq = TB_GET_TO(result);
  unsigned promotes = TB_GET_PROMOTES(result);
  unsigned ep = TB_GET_EP(result);

  chess::Square from(from_sq);
  chess::Square to(to_sq);

  uint16_t type = chess::Move::NORMAL;
  if (ep) {
    type = chess::Move::ENPASSANT;
  } else if (promotes != TB_PROMOTES_NONE) {
    type = chess::Move::PROMOTION;
  } else if (board.at(from).type() == chess::PieceType::KING &&
             std::abs(static_cast<int>(from.file()) - static_cast<int>(to.file())) > 1) {
    type = chess::Move::CASTLING;
    // Fathom gives king destination, but chess-library stores king->rook for castling.
    // For standard chess, rook is on a-file or h-file depending on side.
    to = chess::Square(to.file() > from.file() ? chess::File::FILE_H : chess::File::FILE_A,
                       from.rank());
  }

  chess::Move move;
  if (type == chess::Move::PROMOTION) {
    move =
      chess::Move::make<chess::Move::PROMOTION>(from, to, fathom_promo_to_piece_type(promotes));
  } else if (type == chess::Move::ENPASSANT) {
    move = chess::Move::make<chess::Move::ENPASSANT>(from, to);
  } else if (type == chess::Move::CASTLING) {
    move = chess::Move::make<chess::Move::CASTLING>(from, to);
  } else {
    move = chess::Move::make<chess::Move::NORMAL>(from, to);
  }

  return Move(move.move(), Game::Rules::get_game_phase(board));
}

}  // namespace a0achess
