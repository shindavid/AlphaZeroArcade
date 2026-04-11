#include "games/chess/PolicyEncoding.hpp"

#include "util/Asserts.hpp"

namespace a0achess {

MoveEncodingTable::MoveEncodingTable() {
  int u = 0;
  int v = kNumNonPromoMoves;
  int offset = 0;
  for (int from_sq = 0; from_sq < 64; ++from_sq) {
    int file = from_sq % 8;
    int rank = from_sq / 8;
    uint64_t bitmap = 0;

    // straight lines
    for (int d = 1; d < 8; ++d) {
      add_move(file + d, rank, bitmap);
      add_move(file - d, rank, bitmap);
      add_move(file, rank + d, bitmap);
      add_move(file, rank - d, bitmap);
    }

    // diagonals
    for (int d = 1; d < 8; ++d) {
      add_move(file + d, rank + d, bitmap);
      add_move(file - d, rank + d, bitmap);
      add_move(file + d, rank - d, bitmap);
      add_move(file - d, rank - d, bitmap);
    }

    // knight moves
    auto knight_moves = std::array<std::pair<int, int>, 8>{
      {{2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}}};
    for (const auto& [df, dr] : knight_moves) {
      add_move(file + df, rank + dr, bitmap);
    }

    int8_t promo_offsets[3] = {0, 0, 0};
    if (rank == 6) {
      promo_offsets[0] = 9 * file - 3;
      promo_offsets[1] = 9 * file;
      promo_offsets[2] = 9 * file + 3;
    }

    Data d;
    d.bitmap = bitmap;
    d.offset = offset;
    d.promo_offsets[0] = promo_offsets[0];
    d.promo_offsets[1] = promo_offsets[1];
    d.promo_offsets[2] = promo_offsets[2];

    data_[from_sq] = d;
    offset += std::popcount(bitmap);

    while (bitmap) {
      int to_sq = std::countr_zero(bitmap);
      untyped_move_index_t index = (from_sq << 6) + to_sq;
      untyped_move_indices_[u++] = index;
      bitmap &= (bitmap - 1);  // Clear the lowest set bit

      int to_rank = to_sq / 8;
      if (rank == 6 && to_rank == 7 && to_sq >= from_sq + 7 && to_sq <= from_sq + 9) {
        for (int promo_type_offset = 3; promo_type_offset >= 1; --promo_type_offset) {
          untyped_move_indices_[v++] = index + (promo_type_offset << 12) + chess::Move::PROMOTION;
        }
      }
    }
  }
  RELEASE_ASSERT(offset == kNumNonPromoMoves, "{} != {} @{}", offset, kNumNonPromoMoves, __LINE__);
  RELEASE_ASSERT(u == kNumNonPromoMoves, "{} != {} @{}", u, kNumNonPromoMoves, __LINE__);
  RELEASE_ASSERT(v == kNumMoves, "{} != {} @{}", v, kNumMoves, __LINE__);
}

void MoveEncodingTable::add_move(int to_file, int to_rank, uint64_t& bitmap) {
  if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
    int to_sq = to_rank * 8 + to_file;
    bitmap |= (1ULL << to_sq);
  }
}

}  // namespace a0achess
