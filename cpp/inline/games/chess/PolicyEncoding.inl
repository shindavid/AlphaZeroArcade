#include "games/chess/PolicyEncoding.hpp"
#include "games/chess/Move.hpp"

#include <algorithm>

namespace a0achess {

PolicyEncoding::Index PolicyEncoding::to_index(const InputFrame& frame, const Move& move) {

  chess::Color side_to_move = frame.cur_player == kWhite ? chess::Color::WHITE : chess::Color::BLACK;
  return Index{move_encoding_table.encode(move, side_to_move)};
}

Move PolicyEncoding::to_move(const State& state, const Index& index) {
  return move_encoding_table.decode(index[0], state);
}

MoveEncodingTable::MoveEncodingTable() {
  int offset = 0;
  for (int from_sq = 0; from_sq < 64; ++from_sq) {
    int file = from_sq % 8;
    int rank = from_sq / 8;
    uint64_t bitmap = 0;

    auto add_move = [&](int to_file, int to_rank) {
      if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
        int to_sq = to_rank * 8 + to_file;
        bitmap |= (1ULL << to_sq);
      }
    };

    // straight lines
    for (int d = 1; d < 8; ++d) {
      add_move(file + d, rank);
      add_move(file - d, rank);
      add_move(file, rank + d);
      add_move(file, rank - d);
    }

    // diagonals
    for (int d = 1; d < 8; ++d) {
      add_move(file + d, rank + d);
      add_move(file - d, rank + d);
      add_move(file + d, rank - d);
      add_move(file - d, rank - d);
    }

    // knight moves
    auto knight_moves = std::array<std::pair<int, int>, 8>{
        {{2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}}};
    for (const auto& [df, dr] : knight_moves) {
      add_move(file + df, rank + dr);
    }
    data_[from_sq] = Data{bitmap, offset};
    offset += std::popcount(bitmap);
  }
  num_non_promo_moves_ = offset;
  RELEASE_ASSERT(num_non_promo_moves_ == 1792, "Expected 1792, got {}", num_non_promo_moves_);
}

int MoveEncodingTable::encode(const Move& move, chess::Color side_to_move) const {
  chess::Square from_sq = move.from();
  chess::Square to_sq = move.to();

  if (move.typeOf() == Move::CASTLING) {
    to_sq = chess::Square(to_sq > from_sq ? chess::File::FILE_G : chess::File::FILE_C, from_sq.rank());
  }

  if (side_to_move == chess::Color::BLACK) {
    from_sq = chess::Square(from_sq.index() ^ 56);
    to_sq = chess::Square(to_sq.index() ^ 56);
  }

  const Data& data = data_[from_sq.index()];
  if (move.typeOf() == Move::PROMOTION && move.promotionType() != chess::PieceType::KNIGHT) {
    int df_offset = (int(to_sq.file()) - int(from_sq.file()) + (int(from_sq.file()) > 0)) * 3;
    int pt_offset = int(move.promotionType()) - int(chess::PieceType::BISHOP);
    return num_non_promo_moves_ + promo_bases[from_sq.file()] + df_offset + pt_offset;

  } else {
    return data.offset + count_before_k(data.bitmap, to_sq.index());
  }
}

MoveEncodingTable::MoveData MoveEncodingTable::decode_move_data(int index) const {
  if (index >= num_non_promo_moves_) {
    int promo_idx = index - num_non_promo_moves_;

    auto it = std::upper_bound(promo_bases, promo_bases + 9, promo_idx);
    int from_file = std::distance(promo_bases, it) - 1;

    int rel_idx = promo_idx - promo_bases[from_file];
    int df = (rel_idx / 3) - (from_file > 0);
    int to_file = from_file + df;
    auto pt_enum = chess::PieceType::underlying(
      int(chess::PieceType::BISHOP) + (rel_idx % 3));
    chess::PieceType pt(pt_enum);
    return MoveData{48 + from_file, 56 + to_file, pt};
  }

  auto it = std::upper_bound(data_, data_ + 64, Data{0, index});
  int from_sq = std::distance(data_, it) - 1;
  const Data& data = data_[from_sq];

  int move_rank = index - data.offset;

  uint64_t temp_bitmap = data.bitmap;
  for (int i = 0; i < move_rank; ++i) {
    temp_bitmap &= (temp_bitmap - 1); // Clear the lowest set bit
  }
  int to_sq = std::countr_zero(temp_bitmap);
  return MoveData{from_sq, to_sq, chess::PieceType::NONE};
}

Move MoveEncodingTable::decode(int index, const chess::Board& board) const {
  MoveData move_data = decode_move_data(index);

  chess::Square from_sq(move_data.from_square);
  chess::Square to_sq(move_data.to_square);

  if (board.sideToMove() == chess::Color::BLACK) {
    from_sq = chess::Square(from_sq.index() ^ 56);
    to_sq = chess::Square(to_sq.index() ^ 56);
  }

  auto pt = board.at(from_sq);

  // castling
  if (pt == chess::PieceType::KING && chess::Square::distance(from_sq, to_sq) == 2) {
    to_sq = chess::Square(to_sq > from_sq ? chess::File::FILE_H : chess::File::FILE_A, from_sq.rank());
    return Move::make<Move::CASTLING>(from_sq, to_sq);
  }

  // en passant
  if (pt == chess::PieceType::PAWN && to_sq == board.enpassantSq()) {
    return Move::make<Move::ENPASSANT>(from_sq, to_sq);
  }

  // promotion non-knight
  if (move_data.pt != chess::PieceType::NONE) {
    return Move::make<Move::PROMOTION>(from_sq, to_sq, move_data.pt);
  }

  // promotion knight
  if (pt == chess::PieceType::PAWN && (to_sq.rank() == chess::Rank::RANK_1 || to_sq.rank() == chess::Rank::RANK_8)) {
    return Move::make<Move::PROMOTION>(from_sq, to_sq, chess::PieceType::KNIGHT);
  }

  return Move::make<Move::NORMAL>(from_sq, to_sq);

}

}  // namespace a0achess
