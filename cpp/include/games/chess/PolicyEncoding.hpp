#pragma once

#include "core/BasicTypes.hpp"
#include "games/chess/Constants.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/InputFrame.hpp"
#include "games/chess/Move.hpp"

#include <cstdint>

namespace a0achess {

class MoveEncodingTable {
 public:
  using State = Game::State;

  MoveEncodingTable();

  int encode(const Move& move, core::seat_index_t seat) const;
  chess::Move decode(int index, const chess::Board& board) const;

 private:
  static constexpr int kNumNonPromoMoves = 1792;

  // Disservin's Move uses the following encoding:
  //
  // bits 0-5:   to_square
  // bits 6-11:  from_square
  // bits 12-13: promotion piece type
  // bits 14-15: move type
  //
  // untyped_move_index_t values have bits 0-13 set according to the above encoding. Bits 14-15
  // are set for promotion moves, and otherwise set to 0.
  using untyped_move_index_t = uint16_t;

  struct Data {
    uint64_t bitmap;
    int offset;
    int8_t promo_offsets[3];
  };

  // count number of set bits among the lowest k bits of bitmap
  static int tail_popcount(uint64_t bitmap, int k) {
    return std::popcount(bitmap & ((1ULL << k) - 1));
  }

  static void add_move(int to_file, int to_rank, uint64_t& bitmap);

  MoveEncodingTable(const MoveEncodingTable&) = delete;
  MoveEncodingTable& operator=(const MoveEncodingTable&) = delete;

  Data data_[64];                                         // Indexed by from_square - 1.024KB
  untyped_move_index_t untyped_move_indices_[kNumMoves];  // 3.716KB
};

struct PolicyEncoding {
  using Game = a0achess::Game;
  using State = Game::State;
  using Shape = Eigen::Sizes<kNumMoves>;
  using Tensor = eigen_util::FTensor<Shape>;
  static constexpr int kRank = eigen_util::extract_rank_v<Shape>;
  using Index = Eigen::array<Eigen::Index, kRank>;

  static Index to_index(const InputFrame& frame, const Move& move);
  static Move to_move(const State& state, const Index& index);
  static inline MoveEncodingTable kMoveEncodingTable;
};

}  // namespace a0achess

#include "inline/games/chess/PolicyEncoding.inl"
